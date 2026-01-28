
import os
os.environ['CUDA_NVML_DISABLED'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Phase3AlignmentConfig:
    """Configuration reconstructed from checkpoint"""
    dinov3_model_name: str = "dinov3_vitb16"
    dinov3_feature_dim: int = 768
    eeg_reconstruction_dim: int = 1024  # From learned embeddings
    alignment_hidden_dims: List[int] = None
    alignment_dropout: float = 0.15
    use_cross_attention: bool = True
    num_attention_heads: int = 8
    use_residual_connections: bool = True
    use_layer_norm: bool = True

    def __post_init__(self):
        if self.alignment_hidden_dims is None:
            self.alignment_hidden_dims = [1536, 1024, 768]


class AdvancedDinoV3Encoder(nn.Module):
    """DinoV3 encoder wrapper"""

    def __init__(self, config: Phase3AlignmentConfig):
        super().__init__()
        self.config = config

        # Map config names to actual HuggingFace model identifiers
        model_mapping = {
            "dinov3_vitb14": "facebook/dinov3-vitb14-pretrain-lvd1689m",
            "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "dinov3_vitl14": "facebook/dinov3-vitl14-pretrain-lvd1689m",
            "dinov3_vitg14": "facebook/dinov3-vitg14-pretrain-lvd1689m"
        }

        pretrained_model_name = model_mapping.get(
            config.dinov3_model_name,
            "facebook/dinov3-vitb16-pretrain-lvd1689m"
        )

        logger.info(f"Loading {pretrained_model_name} model...")

        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.dinov3 = AutoModel.from_pretrained(
            pretrained_model_name,
            torch_dtype=torch.float32,
            device_map=None  # Don't use auto device mapping
        )

        for param in self.dinov3.parameters():
            param.requires_grad = False

        self.dinov3.eval()
        logger.info(f''Loaded and froze {pretrained_model_name}")

    def forward(self, images):
        """Extract DinoV3 features"""
        device = images.device

        # Ensure model is on same device as input
        if next(self.dinov3.parameters()).device != device:
            self.dinov3 = self.dinov3.to(device)

        with torch.no_grad():
            outputs = self.dinov3(images)
            # Use pooler_output instead of CLS token
            features = outputs.pooler_output

        return {'features': features}


class AdvancedAlignmentNetwork(nn.Module):
    """Sophisticated alignment network with cross-modal attention-architecture from dino_f.py"""

    def __init__(self, config: Phase3AlignmentConfig):
        super().__init__()
        self.config = config

        # Progressive alignment with residual connections
        alignment_layers = []
        current_dim = config.eeg_reconstruction_dim

        for i, next_dim in enumerate(config.alignment_hidden_dims[1:]):
            alignment_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim) if config.use_layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(config.alignment_dropout)
            ])
            current_dim = next_dim

        # Remove final dropout
        alignment_layers = alignment_layers[:-1]
        self.alignment_projector = nn.Sequential(*alignment_layers)

        # Cross-modal attention mechanism
        if config.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.dinov3_feature_dim,
                num_heads=config.num_attention_heads,
                dropout=config.alignment_dropout,
                batch_first=True
            )

        # Residual connection and output normalization
        self.output_norm = nn.LayerNorm(config.dinov3_feature_dim)
        self.output_dropout = nn.Dropout(config.alignment_dropout * 0.5)  # Lighter dropout at end

    def forward(self, eeg_features: torch.Tensor,
                image_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Align EEG features to DinoV3 feature space

        Args:
            eeg_features: [B, eeg_reconstruction_dim]
            image_features: [B, dinov3_feature_dim] - optional for cross-attention
        """
        batch_size = eeg_features.shape[0]

        # Main alignment projection
        aligned_features = self.alignment_projector(eeg_features)  # [B, dinov3_feature_dim]

        # Cross-modal attention refinement (if image features available)
        if self.config.use_cross_attention and image_features is not None:
            # Use aligned EEG as queries, image features as keys/values
            aligned_features_expanded = aligned_features.unsqueeze(1)  # [B, 1, dinov3_feature_dim]
            image_features_expanded = image_features.unsqueeze(1)  # [B, 1, dinov3_feature_dim]

            attended_features, attention_weights = self.cross_attention(
                query=aligned_features_expanded,
                key=image_features_expanded,
                value=image_features_expanded
            )

            # Residual connection with attended features
            aligned_features = aligned_features + attended_features.squeeze(1)

        # Final normalization and dropout
        aligned_features = self.output_norm(aligned_features)
        aligned_features = self.output_dropout(aligned_features)

        return {'aligned_features': aligned_features}


def load_phase3_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load Phase 3 trained alignment model"""
    logger.info(f"Loading Phase 3 checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct config
    config = Phase3AlignmentConfig()
    if 'config' in checkpoint:
        for key, value in checkpoint['config'].items():
            if hasattr(config, key):
                setattr(config, key, value)

    if config.alignment_hidden_dims is None:
        config.alignment_hidden_dims = [1536, 1024, 768]

    # Initialize models
    dinov3_encoder = AdvancedDinoV3Encoder(config).to(device).float()
    alignment_net = AdvancedAlignmentNetwork(config).to(device).float()

    # Load alignment weights (handle DDP wrapper)
    state_dict = checkpoint['alignment_net_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # alignment_net.load_state_dict(state_dict)
    # Try loading with strict=False first to see what's missing
    try:
        alignment_net.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logger.warning(f"Strict loading failed: {e}")
        logger.info("Trying with strict=False...")
        alignment_net.load_state_dict(state_dict, strict=False)
        logger.warning(" Loaded with strict=False - some weights may be missing")

    # Set to eval
    dinov3_encoder.eval()
    alignment_net.eval()

    for param in dinov3_encoder.parameters():
        param.requires_grad = False
    for param in alignment_net.parameters():
        param.requires_grad = False

    logger.info(f" Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"   Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")

    return dinov3_encoder, alignment_net, config


class LearnedEmbeddingDataset(Dataset):
    """
    Dataset loading LEARNED EEG embeddings from vit_embeddings.py
    These are the PRECOMPUTED 1024-dim embeddings used in dino_f.py training
    """

       def __init__(self, learned_embeddings_dir: Path, images_dir: Path,
                 max_samples: int = None, subjects_to_use: List[str] = None,
                 samples_per_subject: int = None):

        print(f"DEBUG: samples_per_subject = {samples_per_subject}")  # ADD THIS
        print(f"DEBUG: max_samples = {max_samples}")  # ADD THIS
        self.learned_embeddings_dir = learned_embeddings_dir
        self.images_dir = images_dir

        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Build image cache first (for faster lookup)
        logger.info("Building image path cache...")
        self.image_cache = {}
        for img_file in images_dir.rglob("*.jpg"):
            self.image_cache[img_file.name] = img_file
        for img_file in images_dir.rglob("*.png"):
            self.image_cache[img_file.name] = img_file
        logger.info(f"Cached {len(self.image_cache)} images")

        # Check if we should cache images based on count
        if len(self.image_cache) > 50000:
            logger.warning(f"{len(self.image_cache)} images - may use too much RAM")
            logger.warning("Skipping image caching to avoid memory issues")
            self.image_pil_cache = {}  # Empty cache
        else:
            # Preload actual images into RAM
            logger.info("Preloading images into RAM...")
            self.image_pil_cache = {}
            for img_name, img_path in tqdm(self.image_cache.items(), desc="Loading images to RAM"):
                try:
                    self.image_pil_cache[img_name] = Image.open(img_path).convert('RGB')
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")
            logger.info(f" Cached {len(self.image_pil_cache)} PIL images in RAM")



        # # Preload actual images into RAM (not just paths)
        # logger.info("Preloading images into RAM...")
        # self.image_pil_cache = {}
        # for img_name, img_path in tqdm(self.image_cache.items(), desc="Loading images to RAM"):
        #     try:
        #         self.image_pil_cache[img_name] = Image.open(img_path).convert('RGB')
        #     except Exception as e:
        #         logger.warning(f"Failed to load {img_path}: {e}")
        # logger.info(f" Cached {len(self.image_pil_cache)} PIL images in RAM")

        self.samples = []

        # Find subject directories
        subject_dirs = sorted([d for d in learned_embeddings_dir.iterdir()
                               if d.is_dir() and d.name.startswith('sub-')])

        # ADD THIS FILTER:
        if subjects_to_use is not None:
            subject_dirs = [d for d in subject_dirs if d.name in subjects_to_use]
            logger.info(f"Filtered to {len(subject_dirs)} subjects: {[d.name for d in subject_dirs]}")

        logger.info(f"Found {len(subject_dirs)} subject directories")

        # logger.info(f"Found {len(subject_dirs)} subject directories")

        # # Calculate samples per subject
        # if max_samples is not None:
        #     samples_per_subject = max_samples // len(subject_dirs)
        #     logger.info(f"Target: {samples_per_subject} samples per subject")
        # else:
        #     samples_per_subject = None

        # Determine samples per subject
        if samples_per_subject is not None:
            # Use the explicit samples_per_subject parameter
            logger.info(f"Target: {samples_per_subject} samples per subject (explicit)")
        elif max_samples is not None:
            # Calculate from max_samples
            samples_per_subject = max_samples // len(subject_dirs)
            logger.info(f"Target: {samples_per_subject} samples per subject (from max_samples)")
        else:
            # No limit
            samples_per_subject = None
            logger.info(f"No sample limit - loading all available samples")

        for subject_dir in tqdm(subject_dirs, desc="Loading sharded learned embeddings"):
            subject_sample_count = 0
            learned_files = sorted(subject_dir.glob("*_shard_*_learned.pt"))

            if not learned_files:
                logger.warning(f"No learned files found in {subject_dir.name}")
                continue

            for learned_file in learned_files:
                try:
                    data = torch.load(learned_file, weights_only=True)

                    embeddings = data['learned_features']
                    labels = data['labels']
                    image_filenames = data['image_filenames']
                    super_categories = data.get('super_categories', ['unknown'] * len(embeddings))
                    subject_id = data['subject_id']

                    valid_mask = labels >= 0
                    n_valid = valid_mask.sum().item()

                    if n_valid == 0:
                        continue

                    embeddings = embeddings[valid_mask]
                    valid_filenames = [image_filenames[i] for i in range(len(image_filenames)) if valid_mask[i]]
                    valid_categories = [super_categories[i] for i in range(len(super_categories)) if valid_mask[i]]

                    for i in range(len(embeddings)):
                        # Stop when this subject reaches its quota
                        if samples_per_subject is not None and subject_sample_count >= samples_per_subject:
                            break

                        img_filename = valid_filenames[i]

                        if img_filename in self.image_pil_cache:
                            self.samples.append({
                                'learned_embedding': embeddings[i].numpy(),
                                'subject_id': subject_id,
                                'dataset': 'learned',
                                'image_path': img_filename,
                                'image_pil': self.image_pil_cache[img_filename],
                                'category': valid_categories[i]
                            })
                            subject_sample_count += 1
                        else:
                            img_path = self.image_cache.get(img_filename)
                            if img_path and img_path.exists():
                                self.samples.append({
                                    'learned_embedding': embeddings[i].numpy(),
                                    'subject_id': subject_id,
                                    'dataset': 'learned',
                                    'image_path': str(img_path),
                                    'category': valid_categories[i]
                                })
                                subject_sample_count += 1

                    # Break from shard loop if subject quota reached
                    if samples_per_subject is not None and subject_sample_count >= samples_per_subject:
                        break

                except Exception as e:
                    logger.warning(f"Error loading {learned_file}: {e}")
                    continue

            logger.info(f"{subject_id}: collected {subject_sample_count} samples")

        # Log final distribution
        from collections import Counter
        subject_counts = Counter([s['subject_id'] for s in self.samples])
        logger.info(f"Final distribution: {dict(subject_counts)}")
        logger.info(f" Loaded {len(self.samples)} learned embeddings from shards")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get learned embedding (already processed through EEG model)
        learned_embedding = torch.from_numpy(sample['learned_embedding']).float()

        # Load and transform image
        # image_tensor = None
        # if sample['image_path']:
        #     img_path = Path(sample['image_path'])
        #     if not img_path.exists():
        #         img_path = self.images_dir / img_path.name
        #
        #     if img_path.exists():
        #         try:
        #             image = Image.open(img_path).convert('RGB')
        #             image_tensor = self.image_transform(image)
        #         except Exception as e:
        #             logger.warning(f"Error loading image: {e}")

        # Load and transform image (use cached version)
        image_tensor = None
        if sample['image_path']:
            img_filename = Path(sample['image_path']).name



            # # Try to use cached PIL image first
            # if img_filename in self.image_pil_cache:
            #     try:
            #         image_tensor = self.image_transform(self.image_pil_cache[img_filename])
            #     except Exception as e:
            #         logger.warning(f"Error transforming cached image: {e}")

            if 'image_pil' in sample:
                try:
                    image_tensor = self.image_transform(sample['image_pil'])
                except Exception as e:
                    logger.warning(f"Error transforming preloaded image: {e}")
            elif sample.get('image_path'):


                # Fallback: load from disk if not cached
                img_path = Path(sample['image_path'])
                if not img_path.exists():
                    img_path = self.images_dir / img_path.name

                if img_path.exists():
                    try:
                        image = Image.open(img_path).convert('RGB')
                        image_tensor = self.image_transform(image)
                    except Exception as e:
                        logger.warning(f"Error loading image: {e}")

        return {
            # 'learned_embedding': learned_embedding,
            'learned_embedding': torch.from_numpy(sample['learned_embedding']).float(),  # Already tensor
            'image': image_tensor,
            'subject_id': sample['subject_id'],
            'dataset': sample['dataset'],
            'category': sample['category'],
            'image_path': sample['image_path']
        }


def extract_aligned_embeddings(
        phase3_checkpoint: str,
        learned_embeddings_dir: str,
        images_dir: str,
        output_path: str,
        max_samples: int = 2000,
        batch_size: int = 256,  # CHANGED: Much larger for inference
        subjects_to_use=None,
        samples_per_subject=None,
        device: str = 'cuda',
        # NEW PERFORMANCE PARAMETERS
        num_workers: int = 8,  # Parallel data loading
        prefetch_factor: int = 4,  # Prefetch batches
        use_amp: bool = True,  # Mixed precision
):
    """
   Extract aligned embeddings using fast inference techniques
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load trained alignment model
    dinov3_encoder, alignment_net, config = load_phase3_checkpoint(phase3_checkpoint, device)

    # Set to eval and disable gradients
    dinov3_encoder.eval()
    alignment_net.eval()

    for param in dinov3_encoder.parameters():
        param.requires_grad = False
    for param in alignment_net.parameters():
        param.requires_grad = False

    # Create dataset
    dataset = LearnedEmbeddingDataset(
        learned_embeddings_dir=Path(learned_embeddings_dir),
        images_dir=Path(images_dir),
        max_samples=max_samples,
        subjects_to_use=subjects_to_use,
        samples_per_subject=samples_per_subject
    )

    #  DATALOADER
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Much larger
        shuffle=False,  # No shuffle needed for inference
        num_workers=num_workers,  # Parallel loading
        pin_memory=True,  # Faster CPU->GPU
        prefetch_factor=prefetch_factor,  # Prefetch batches
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
        drop_last=False
    )

    # Storage
    aligned_embeddings = []
    learned_embeddings = []
    subject_ids = []
    dataset_names = []
    categories = []
    image_paths = []

    logger.info("Extracting aligned embeddings (learned_emb → alignment)...")

    # INFERENCE LOOP
    with torch.no_grad():  # Disable gradient computation
        # Use mixed precision if available
        autocast_context = torch.cuda.amp.autocast() if use_amp and device.type == 'cuda' else torch.no_grad()

        with autocast_context:
            for batch in tqdm(dataloader, desc="Processing batches"):
                # Filter valid samples (with images)
                valid_idx = [i for i, img in enumerate(batch['image']) if img is not None]
                if not valid_idx:
                    continue

                # Move to device with non_blocking for async transfer
                learned_emb = batch['learned_embedding'][valid_idx].to(device, non_blocking=True).float()
                images = torch.stack([batch['image'][i] for i in valid_idx]).to(device, non_blocking=True).float()

                # Extract DinoV3 features (using dummy for diffusion)
                image_features = torch.zeros(len(learned_emb), 768, device=device).float()

                # Align learned embeddings
                alignment_outputs = alignment_net(learned_emb, image_features)
                aligned = alignment_outputs['aligned_features'].float()

                # Move to CPU and store (batch operation)
                aligned_embeddings.append(aligned.cpu().numpy())
                learned_embeddings.append(learned_emb.cpu().numpy())

                # Store metadata
                for i in valid_idx:
                    subject_ids.append(batch['subject_id'][i])
                    dataset_names.append(batch['dataset'][i])
                    categories.append(batch['category'][i] if batch['category'][i] else 'unknown')
                    image_paths.append(batch['image_path'][i] if batch['image_path'][i] else '')

    # Concatenate all
    aligned_embeddings = np.vstack(aligned_embeddings)
    learned_embeddings = np.vstack(learned_embeddings)

    logger.info(f" Extracted {len(aligned_embeddings)} samples")
    logger.info(f"   Aligned embeddings: {aligned_embeddings.shape}")
    logger.info(f"   Learned embeddings: {learned_embeddings.shape}")

    # Save to HDF5
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('aligned_embeddings', data=aligned_embeddings, compression='gzip')
        f.create_dataset('learned_embeddings', data=learned_embeddings, compression='gzip')
        f.create_dataset('subject_ids', data=np.array(subject_ids, dtype='S'))
        f.create_dataset('dataset_names', data=np.array(dataset_names, dtype='S'))
        f.create_dataset('categories', data=np.array(categories, dtype='S'))
        f.create_dataset('image_paths', data=np.array(image_paths, dtype='S'))

        f.attrs['phase3_checkpoint'] = phase3_checkpoint
        f.attrs['n_samples'] = len(aligned_embeddings)
        f.attrs['aligned_dim'] = aligned_embeddings.shape[1]
        f.attrs['config'] = json.dumps(config.__dict__, default=str)

    logger.info(f" Saved to {output_path}")

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE - READY FOR DIFFUSION TRAINING")
    print("=" * 70)
    print(f"Total samples: {len(aligned_embeddings)}")
    print(f"Aligned embeddings: {aligned_embeddings.shape} → for diffusion conditioning")
    print(f"Image paths: {len(image_paths)} → for diffusion targets")
    print(f"\nPipeline verified:")
    print(f" Learned embeddings (precomputed) → Alignment → Aligned embeddings")
    print("=" * 70 + "\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract aligned embeddings from Phase 3 DINO checkpoint"
    )

    parser.add_argument('--phase3_checkpoint', type=str, required=True,
                        help='Phase 3 alignment checkpoint (.pth)')
    parser.add_argument('--learned_embeddings_dir', type=str, required=True,
                        help='Directory with learned embeddings from vit_embeddings.py')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Images directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output HDF5 file path')
    # parser.add_argument('--max_samples', type=int, default=10000,
    #                     help='Max samples to extract')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--max_samples', type=int, default=None,  # Changed from default=2000
                        help='Max samples to extract (None = all)')

    parser.add_argument('--subjects_to_use', type=str, nargs='+', default=None,
                        help='Specific subject IDs to extract (e.g., sub-01 sub-02)')
    parser.add_argument('--samples_per_subject', type=int, default=None,
                        help='Number of samples per subject (if not set, uses max_samples/num_subjects)')


    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='Number of batches to prefetch')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision (FP16)')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                        help='Disable automatic mixed precision')

    args = parser.parse_args()

    extract_aligned_embeddings(
        phase3_checkpoint=args.phase3_checkpoint,
        learned_embeddings_dir=args.learned_embeddings_dir,
        images_dir=args.images_dir,
        output_path=args.output_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        subjects_to_use=args.subjects_to_use,
        samples_per_subject=args.samples_per_subject,
        device=args.device,
        num_workers=args.num_workers,  # I added
        prefetch_factor=args.prefetch_factor,  # I added
        use_amp=args.use_amp)  # NEW



    # extract_aligned_embeddings(
    #     phase3_checkpoint=args.phase3_checkpoint,
    #     learned_embeddings_dir=args.learned_embeddings_dir,
    #     images_dir=args.images_dir,
    #     output_path=args.output_path,
    #     max_samples=args.max_samples,
    #     batch_size=args.batch_size,
    #     subjects_to_use=args.subjects_to_use,  # ADD THIS
    #     samples_per_subject=args.samples_per_subject,  # ADD THIS
    #     device=args.device
    # )


if __name__ == "__main__":
    main()