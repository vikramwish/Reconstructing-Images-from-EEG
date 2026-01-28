
# Core PyTorch imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms

# Scientific computing
import numpy as np
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns

# Data handling
import h5py
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import argparse
import time
import os
from tqdm import tqdm
import warnings
from PIL import Image
import random
import signal
import sys
# System imports

import traceback
from collections import defaultdict, Counter

from transformers import AutoImageProcessor, AutoModel

from transformers import CLIPProcessor, CLIPModel

# Import the EEGViTCNet from your eeg_vit.py
try:
    from eeg_vit import EEGViTCNetSelfAware, EEGViTCNetConfig
except ImportError as e:
    print(f"Warning: Could not import EEG model classes: {e}")
    print("Make sure eeg_vit.py is in your Python path")


    # Define dummy classes to prevent import errors during testing
    class EEGViTCNetSelfAware:
        pass


    class EEGViTCNetConfig:
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class Phase3AlignmentConfig:
    """Configuration for Phase 3 EEG-DinoV3 alignment"""

    # Model Architecture - updated for HuggingFace
    dinov3_model_name: str = "dinov3_vitb16"  # Changed from vitb14 to vitb16
    dinov3_feature_dim: int = 768  # This should match the model's output dim
    eeg_reconstruction_dim: int = 1024  # From EEGViTCNet reconstruction output
    # Addiing new fields:
    use_clip_text: bool = False
    clip_weight: float = 0.3
    holdout_ratio: float = 0.2

    # Alignment Network Architecture - Progressive complexity
    alignment_hidden_dims: List[int] = None
    alignment_dropout: float = 0.15
    use_cross_attention: bool = True
    num_attention_heads: int = 8
    use_residual_connections: bool = True
    use_layer_norm: bool = True

    # Multimodal Learning Techniques
    use_contrastive_learning: bool = True
    use_triplet_loss: bool = True
    use_adversarial_alignment: bool = False  # Can destabilise early training
    contrastive_temperature: float = 0.07
    triplet_margin: float = 0.2

    # Loss Function Weights - Carefully tuned for stability
    mse_weight: float = 1.0
    contrastive_weight: float = 0.5  # Start lower, increase gradually
    triplet_weight: float = 0.3
    consistency_weight: float = 0.4
    perceptual_weight: float = 0.6  # DinoV3 internal similarity
    category_consistency_weight: float = 0.2  # MoE routing consistency

    # Training Parameters - Conservative for stability
    batch_size: int = 96  # Reduced for memory efficiency
    learning_rate: float = 5e-5  # Lower for fine-grained alignment
    # learning_rate: float = 8e-5 # cahnged
    weight_decay: float = 1e-5
    num_epochs: int = 150  # More epochs for gradual learning
    warmup_epochs: int = 5
    scheduler_patience: int = 20 # changed from 20
    gradient_clip_norm: float = 1.0
    # lr_scale_on_plateau: bool = True

    # Data Parameters
    image_size: int = 224
    max_trials_per_subject: int = 150
    train_split: float = 0.8

    # Multi-GPU Training
    use_distributed: bool = True
    sync_bn: bool = True
    find_unused_parameters: bool = False

    # Visualization and Analysis Parameters
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    visualization_samples: int = 2500
    save_visualizations_every: int = 25  # epochs

    def __post_init__(self):
        if self.alignment_hidden_dims is None:
            # Progressive dimension reduction for smooth alignment
            self.alignment_hidden_dims = [
                self.eeg_reconstruction_dim,
                1024,
                896,
                self.dinov3_feature_dim
            ]



class AdvancedDinoV3Encoder(nn.Module):
    """DinoV3 encoder using HuggingFace transformers"""

    def __init__(self, config: Phase3AlignmentConfig):
        super().__init__()
        self.config = config
        # self.device = config.device
        # self.device = None

        # Map your config names to HuggingFace model names
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

        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.dinov3 = AutoModel.from_pretrained(
            pretrained_model_name,
            # device_map="auto" if torch.cuda.is_available() else None
            device_map = None
        )

        # Freeze DinoV3
        for param in self.dinov3.parameters():
            param.requires_grad = False

        self.dinov3.eval()
        logger.info(f"✓ Loaded and froze {pretrained_model_name}")

        # ADD: Optional CLIP text encoder
        if config.use_clip_text:
            # self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(config.device)
            # self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_safetensors=True  # Force safetensors format
            )
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip.eval()
            logger.info("CLIP text encoder initialized")
            # ADDED
            for param in self.clip.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract hierarchical DinoV3 features"""
        device = images.device
        with torch.no_grad():
            # Convert tensor back to PIL format for processor
            # images shape: [B, 3, 224, 224] with values in [0,1] range after normalization

            # Denormalize images (reverse ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            denorm_images = images * std + mean
            denorm_images = torch.clamp(denorm_images, 0, 1)

            # Process images through the HuggingFace processor format
            # Since we already have tensors, we can create the input dict directly
            # Ensure dinov3 is on same device
            if next(self.dinov3.parameters()).device != device:
                self.dinov3 = self.dinov3.to(device)

            # inputs = {
            #     'pixel_values': denorm_images.to(self.dinov3.device)
            # }
            inputs = {'pixel_values': denorm_images}
            outputs = self.dinov3(**inputs)
            features = outputs.pooler_output


            # Get features using the HuggingFace model
            outputs = self.dinov3(**inputs)
            features = outputs.pooler_output  # [B, feature_dim]

        return {
            'features': features,
            'normalized_features': F.normalize(features, dim=-1)
        }

    def forward_text(self, texts):
        if not self.config.use_clip_text or texts is None or self.clip is None:
            return None

        # Get device from the CLIP model itself
        device = next(self.clip.parameters()).device
        tokens = self.clip_processor(text=texts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():  # Add this for efficiency
            return self.clip.get_text_features(**tokens)

class AdvancedAlignmentNetwork(nn.Module):
    """Sophisticated alignment network with cross-modal attention"""

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
                aligned_features_expanded,
                image_features_expanded,
                image_features_expanded
            )
            aligned_features = attended_features.squeeze(1)  # [B, dinov3_feature_dim]
        else:
            attention_weights = None

        # Final normalization and residual connection
        if self.config.use_residual_connections and aligned_features.shape == eeg_features.shape:
            # Only if dimensions match
            aligned_features = aligned_features + eeg_features

        aligned_features = self.output_norm(aligned_features)
        aligned_features = self.output_dropout(aligned_features)

        return {
            'aligned_features': aligned_features,
            'normalized_aligned': F.normalize(aligned_features, dim=-1),
            'attention_weights': attention_weights
        }
    # def forward(self, eeg_features: torch.Tensor,
    #             image_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    #     """Align EEG features to DinoV3 feature space"""
    #
    #     batch_size = eeg_features.shape[0]
    #
    #     # Main alignment projection
    #     aligned_features = self.alignment_projector(eeg_features)  # [B, dinov3_feature_dim]
    #
    #     # TEMPORARILY SKIP cross-attention to debug
    #     # if self.config.use_cross_attention and image_features is not None:
    #     #     aligned_features_expanded = aligned_features.unsqueeze(1)
    #     #     image_features_expanded = image_features.unsqueeze(1)
    #     #     attended_features, attention_weights = self.cross_attention(...)
    #     #     aligned_features = attended_features.squeeze(1)
    #
    #     attention_weights = None  # Skip for now
    #
    #     # Final normalization
    #     aligned_features = self.output_norm(aligned_features)
    #     aligned_features = self.output_dropout(aligned_features)
    #
    #     return {
    #         'aligned_features': aligned_features,
    #         'normalized_aligned': F.normalize(aligned_features, dim=-1),
    #         'attention_weights': attention_weights
    #     }
    #


class MultimodalAlignmentLoss(nn.Module):
    def __init__(self, config: Phase3AlignmentConfig):
        super().__init__()
        self.config = config
        self.temp = config.contrastive_temperature  # Use existing config field

        # Add projection for CLIP text (512) -> DinoV3 space (768)
        if config.use_clip_text:
            self.text_projection = nn.Linear(512, 768)

    # def forward(self, eeg_features, image_features, text_features=None, routing_probs=None):
    #     losses = {}
    #
    #     # Normalize
    #     eeg_norm = F.normalize(eeg_features, dim=-1)
    #     img_norm = F.normalize(image_features, dim=-1)
    #
    #     # EEG ↔ DINO contrastive
    #     logits = eeg_norm @ img_norm.T / self.temp
    #     labels = torch.arange(len(eeg_features), device=eeg_features.device)
    #     losses['contrastive_dino'] = F.cross_entropy(logits, labels)  # Rename for clarity
    #     losses['total'] = losses['contrastive_dino']
    #
    #     # Calculate actual MSE for logging
    #     losses['mse'] = F.mse_loss(eeg_features, image_features)
    #
    #     # EEG ↔ CLIP text contrastive
    #     if self.config.use_clip_text and text_features is not None:
    #         text_projected = self.text_projection(text_features)
    #         txt_norm = F.normalize(text_projected, dim=-1)
    #         logits_text = eeg_norm @ txt_norm.T / self.temp
    #         losses['contrastive_clip'] = self.config.clip_weight * F.cross_entropy(logits_text, labels)
    #         losses['total'] += losses['contrastive_clip']
    #
    #     return losses

    def forward(self, eeg_features, image_features, text_features=None, routing_probs=None):
        losses = {}

        eeg_norm = F.normalize(eeg_features, dim=-1)
        img_norm = F.normalize(image_features, dim=-1)

        # Symmetric contrastive loss (like CLIP)
        logits_eeg_to_img = eeg_norm @ img_norm.T / self.temp
        logits_img_to_eeg = img_norm @ eeg_norm.T / self.temp

        labels = torch.arange(len(eeg_features), device=eeg_features.device)

        loss_e2i = F.cross_entropy(logits_eeg_to_img, labels)
        loss_i2e = F.cross_entropy(logits_img_to_eeg, labels)

        losses['contrastive_dino'] = (loss_e2i + loss_i2e) / 2  # Symmetric
        losses['total'] = losses['contrastive_dino']

        # Keep MSE for monitoring only
        losses['mse'] = F.mse_loss(eeg_features, image_features)

        # CLIP text loss (keep as is)
        if self.config.use_clip_text and text_features is not None:
            text_projected = self.text_projection(text_features)
            txt_norm = F.normalize(text_projected, dim=-1)

            logits_eeg_to_txt = eeg_norm @ txt_norm.T / self.temp
            logits_txt_to_eeg = txt_norm @ eeg_norm.T / self.temp

            loss_e2t = F.cross_entropy(logits_eeg_to_txt, labels)
            loss_t2e = F.cross_entropy(logits_txt_to_eeg, labels)

            losses['contrastive_clip'] = self.config.clip_weight * (loss_e2t + loss_t2e) / 2
            losses['total'] += losses['contrastive_clip']

        return losses

class Phase3EEGImageDataset(Dataset):
    """Dataset that loads pre-extracted 1024D EEG embeddings with image filenames"""

    def __init__(self, extracted_pairs: List[Dict], config: Phase3AlignmentConfig, split: str = 'train'):
        self.data_pairs = extracted_pairs
        self.config = config
        self.split = split

        # Image preprocessing (keep existing transform logic)
        self.image_transform = self._get_image_transform()

        logger.info(f"Created {split} dataset with {len(extracted_pairs)} pairs")

    def _get_image_transform(self):
        # Keep your existing transform logic
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        pair = self.data_pairs[idx]

        try:
            # Load pre-extracted EEG embedding (1024D)
            eeg_tensor = pair['eeg_embedding']  # Already a tensor

            # Load image
            image = Image.open(pair['image_path']).convert('RGB')
            image_tensor = self.image_transform(image)

            # Text caption for CLIP
            text_caption = None
            if self.config.use_clip_text:
                text_caption = pair.get('super_category', 'unknown')

            return {
                'eeg_embedding': eeg_tensor,  # [1024]
                'image': image_tensor,  # [3, 224, 224]
                'subject_id': pair['subject_id'],
                'super_categories': pair['super_category'],
                # 'image_path': pair['image_path'],
                'image_path': str(pair['image_path']),
                'text_caption': text_caption,
                'routing_probs': pair.get('routing_probs', torch.zeros(4))
            }

        except Exception as e:
            logger.warning(f"Failed to load sample {idx}: {e}")
            # Return dummy sample
            return {
                'eeg_embedding': torch.zeros(1024),
                'image': torch.zeros(3, self.config.image_size, self.config.image_size),
                'subject_id': 'unknown',
                'super_categories': 'unknown',
                'image_path': '',
                'text_caption': 'unknown' if self.config.use_clip_text else None,
                'routing_probs': torch.zeros(4)
            }

def create_holdout_split(data_pairs, holdout_ratio=0.2):
    """Create hold-out split stratified by category"""
    from collections import defaultdict
    import random

    # Group by category
    category_groups = defaultdict(list)
    for idx, pair in enumerate(data_pairs):
        category = pair.get('super_category', 'unknown')
        category_groups[category].append(idx)

    train_indices = []
    test_indices = []

    for category, indices in category_groups.items():
        random.shuffle(indices)
        split_point = int(len(indices) * (1 - holdout_ratio))
        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    logger.info(f"Hold-out split: {len(train_indices)} train, {len(test_indices)} test")
    return train_indices, test_indices


def compute_retrieval_metrics(eeg_features, target_features, k_values=[1, 5]):
    """Compute top-k retrieval accuracy"""
    eeg_norm = F.normalize(eeg_features, dim=-1)
    target_norm = F.normalize(target_features, dim=-1)

    similarities = eeg_norm @ target_norm.T
    ranks = torch.argsort(similarities, dim=1, descending=True)
    correct_ranks = torch.diagonal(ranks).cpu().numpy()

    metrics = {}
    for k in k_values:
        top_k_acc = np.mean(correct_ranks < k)
        metrics[f'top_{k}'] = top_k_acc

    return metrics


def evaluate_retrieval(model, encoder, test_loader, device, config):
    """Evaluate EEG->Image and EEG->Text retrieval"""
    model.eval()
    encoder.eval()

    eeg_features = []
    image_features = []
    text_features = []

    with torch.no_grad():
        for batch in test_loader:
            # eeg_feats = model(batch['eeg_embedding'].to(device))
            alignment_outputs = model(batch['eeg_embedding'].to(device))
            eeg_feats = alignment_outputs['aligned_features']
            eeg_features.append(eeg_feats)

            img_outputs = encoder(batch['image'].to(device))
            img_feats = img_outputs['features']
            # img_feats = encoder(batch['image'].to(device))
            image_features.append(img_feats)

            if config.use_clip_text and batch['text_caption'][0] is not None:
                txt_feats = encoder.forward_text(batch['text_caption'])
                text_features.append(txt_feats)

    eeg_features = torch.cat(eeg_features, dim=0)
    image_features = torch.cat(image_features, dim=0)

    img_metrics = compute_retrieval_metrics(eeg_features, image_features)
    results = {'eeg_to_image': img_metrics}

    if text_features:
        text_features = torch.cat(text_features, dim=0)
        txt_metrics = compute_retrieval_metrics(eeg_features, text_features)
        results['eeg_to_text'] = txt_metrics

    return results

class Phase3AlignmentTrainer:
    """Unified trainer for Phase 3 EEG-DinoV3 alignment with combined subject training"""

    def __init__(self, config: Phase3AlignmentConfig,
                 extracted_embeddings_dir: str, images_dir: str, output_dir: str):

        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.should_stop = False
        # Add signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Initialize distributed training if available (keep existing logic)
        if config.use_distributed and torch.cuda.device_count() > 1:
            self._init_distributed()
        else:
            self.local_rank = 0
            self.world_size = 1

        self.device = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'

        # Remove EEG model loading - we use pre-extracted embeddings
        # self.eeg_model = self._load_eeg_model(eeg_model_path)  # REMOVE THIS

        # Initialize models (keep existing)
        self._build_models()

        # Create datasets from extracted embeddings
        self._create_combined_datasets(extracted_embeddings_dir, images_dir)

        # Initialize training (keep existing)
        self._init_training()

        # Training state (keep existing)
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def _signal_handler(self, signum, frame):
        logger.info("Received interrupt signal. Stopping training...")
        self.should_stop = True

    def _init_distributed(self):
        """Initialize distributed training"""
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.local_rank)

        logger.info(f" Initialized distributed training: rank {self.local_rank}/{self.world_size}")

    def _load_eeg_model(self, model_path: str) -> EEGViTCNetSelfAware:
        """Load pre-trained EEGViTCNet model"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Reconstruct config
        eeg_config = EEGViTCNetConfig()
        if 'config' in checkpoint:
            for key, value in checkpoint['config'].items():
                if hasattr(eeg_config, key):
                    setattr(eeg_config, key, value)

        # Create and load model
        model = EEGViTCNetSelfAware(eeg_config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Freeze EEG model - we only use it for feature extraction
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        logger.info(f"Loaded pre-trained EEG model from {model_path}")
        return model

    def _build_models(self):
        """Build alignment models"""
        # DinoV3 encoder
        device = torch.device(f'cuda:{self.local_rank}')
        self.dinov3_encoder = AdvancedDinoV3Encoder(self.config).to(self.device)
        if hasattr(self.dinov3_encoder, 'clip'):
            self.dinov3_encoder.clip = self.dinov3_encoder.clip.to(device)

        # Alignment network
        self.alignment_net = AdvancedAlignmentNetwork(self.config).to(self.device)

        # Loss function
        self.criterion = MultimodalAlignmentLoss(self.config).to(self.device)

        # Wrap with DDP if distributed
        if self.config.use_distributed and self.world_size > 1:
            self.alignment_net = DDP(
                self.alignment_net,
                device_ids=[self.local_rank],
                output_device=self.local_rank,  #
                find_unused_parameters=self.config.find_unused_parameters
            )

        logger.info(" Built alignment models successfully on cuda:{self.local_rank}")

       #     logger.info(f"Created combined datasets: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
    def _create_combined_datasets(self, extracted_embeddings_dir: str, images_dir: str):
        """Create combined datasets from extracted embeddings"""
        logger.info("Creating datasets from extracted embeddings...")

        # Load all extracted pairs
        all_pairs = self._load_extracted_pairs(extracted_embeddings_dir, images_dir)

        if not all_pairs:
            raise ValueError("No valid EEG-image pairs found!")

        # Shuffle and split
        random.shuffle(all_pairs)
        split_idx = int(len(all_pairs) * self.config.train_split)
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]

        logger.info(f"Total pairs: {len(all_pairs)}")
        logger.info(f"Train/Val split: {len(train_pairs)}/{len(val_pairs)}")

        # Create datasets (no EEG model needed)
        self.train_dataset = Phase3EEGImageDataset(train_pairs, self.config, split='train')
        self.val_dataset = Phase3EEGImageDataset(val_pairs, self.config, split='val')

        # Create data loaders (keep existing logic)
        if self.config.use_distributed and self.world_size > 1:
            self.train_sampler = DistributedSampler(self.train_dataset)
            self.val_sampler = DistributedSampler(self.val_dataset)
        else:
            self.train_sampler = None
            self.val_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            num_workers=32,
            prefetch_factor=4,
            persistent_workers=True,  # Add this - prevents hanging
            pin_memory=True,
            drop_last=True

        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=16,
            prefetch_factor=2,
            pin_memory=True,
            drop_last=False
        )

        logger.info(f"✓ Created datasets: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
       def _load_extracted_pairs(self, extracted_embeddings_dir: str, images_dir: str) -> List[Dict]:
        """Load EEG-image pairs from extracted embedding shards"""
        extracted_dir = Path(extracted_embeddings_dir)
        images_root = Path(images_dir)

        # PRE-BUILD IMAGE PATH CACHE - This is the key optimization
        # logger.info("Building image path cache...")
        logger.info("Caching image paths...")
        image_cache = {}
        for img_file in images_root.rglob("*.jpg"):  # Adjust extensions as needed
            image_cache[img_file.name] = img_file
        for img_file in images_root.rglob("*.png"):
            image_cache[img_file.name] = img_file
        logger.info(f"Cached {len(image_cache)} images")

        all_pairs = []
        subject_dirs = sorted([d for d in extracted_dir.iterdir()
                               if d.is_dir() and d.name.startswith('sub-')])

        for subject_dir in tqdm(subject_dirs, desc="Loading subjects"):
            learned_files = sorted(subject_dir.glob("*_shard_*_learned.pt"))

            if not learned_files:
                continue

            for learned_file in learned_files:
                try:
                    data = torch.load(learned_file, weights_only=True)

                    embeddings = data['learned_features']
                    labels = data['labels']
                    routing_probs = data['routing_probs']
                    image_filenames = data['image_filenames']
                    super_categories = data['super_categories']
                    subject_id = data['subject_id']

                    valid_mask = labels >= 0
                    n_valid = valid_mask.sum().item()

                    if n_valid == 0:
                        continue

                    # Apply mask
                    embeddings = embeddings[valid_mask]
                    labels = labels[valid_mask]
                    routing_probs = routing_probs[valid_mask]
                    valid_filenames = [image_filenames[i] for i in range(len(image_filenames)) if valid_mask[i]]
                    valid_super_cats = [super_categories[i] for i in range(len(super_categories)) if valid_mask[i]]

                    # FAST IMAGE PATH LOOKUP
                    for i in range(len(embeddings)):
                        filename = valid_filenames[i]

                        # Quick cache lookup instead of filesystem search
                        if filename in image_cache:
                            image_path = image_cache[filename]
                        else:
                            logger.warning(f"Image not found in cache: {filename}")
                            continue

                        pair = {
                            'eeg_embedding': embeddings[i],
                            'image_path': image_path,
                            'image_filename': filename,
                            'super_category': valid_super_cats[i],
                            'subject_id': subject_id,
                            'routing_probs': routing_probs[i],
                            'label': labels[i].item()
                        }
                        all_pairs.append(pair)

                except Exception as e:
                    logger.error(f"Error loading {learned_file}: {e}")
                    continue

        return all_pairs




    def _find_image_path(self, filename: str, images_dir: str) -> Optional[Path]:
        """Find the full path to an image given its filename"""
        images_root = Path(images_dir)

        # Try different possible locations
        possible_paths = [
            images_root / filename,
            images_root / "stimuli" / filename,
            images_root / "images" / filename
        ]

        # Also try nested directories
        for subdir in images_root.rglob("*/"):
            possible_paths.append(subdir / filename)

        for path in possible_paths:
            if path.exists():
                return path

        return None



    def _infer_dataset_from_path(self, file_path: Path) -> str:
        """Infer dataset name from file path"""
        path_str = str(file_path).lower()
        if 'alljoined' in path_str or 'fm' in path_str:
            return 'fm_alljoined'
        elif 'infant' in path_str or 'ds005106' in path_str:
            return 'infant_ds005106'
        return 'unknown'

    def _init_training(self):
        """Initialize training components"""
        # Optimizer - only train alignment network
        self.optimizer = torch.optim.AdamW(
            self.alignment_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.scheduler_patience,
            verbose=True
        )

        # Warmup scheduler
        # self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     self.optimizer,
        #     start_factor=0.01,
        #     end_factor=1.0,
        #     total_iters=self.config.warmup_epochs
        # )

        logger.info(" Initialized training components")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training"""

        # checkpoint_path =
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if hasattr(self.alignment_net, 'module'):
            self.alignment_net.module.load_state_dict(checkpoint['alignment_net_state_dict'])
        else:
            self.alignment_net.load_state_dict(checkpoint['alignment_net_state_dict'])

        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
        logger.info(f"Best val loss so far: {self.best_val_loss:.4f}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        print(f"[Rank {self.local_rank}] Devices:")
        print(f"  Alignment net: {next(self.alignment_net.parameters()).device}")
        print(f"  DinoV3: {next(self.dinov3_encoder.dinov3.parameters()).device}")
        if hasattr(self.dinov3_encoder, 'clip'):
            print(f"  CLIP: {next(self.dinov3_encoder.clip.parameters()).device}")

        self.alignment_net.train()

        epoch_losses = defaultdict(list)

        # Set epoch for distributed sampler
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch + 1}",
                    disable=(self.local_rank != 0))  # Update every 30 seconds max)

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            eeg_embeddings = batch['eeg_embedding'].to(self.device)
            images = batch['image'].to(self.device)
            if self.should_stop:
                logger.info("Training interrupted by user")
                return {'total': 0.0, 'mse': 0.0}
            # DEBUG: Print actual shapes
            # print(f"DEBUG - Batch {batch_idx}:")
            # print(f"  EEG embeddings shape: {eeg_embeddings.shape}")
            # print(f"  Images shape: {images.shape}")


            routing_probs = batch.get('routing_probs', None)
            if routing_probs is not None:
                routing_probs = routing_probs.to(self.device)

            # Extract DinoV3 features
            with torch.no_grad():
                dinov3_outputs = self.dinov3_encoder(images)
                image_features = dinov3_outputs['features']
                # print(f"  DinoV3 features shape: {image_features.shape}")

                # ADD: Get text features if enabled
                text_features = None
                if self.config.use_clip_text and 'text_caption' in batch:
                    text_features = self.dinov3_encoder.forward_text(batch['text_caption'])

            # Forward pass through alignment network
            #alignment_outputs = self.alignment_net(eeg_embeddings, image_features)
            #aligned_features = alignment_outputs['aligned_features']
            # Forward pass through alignment network
            alignment_outputs = self.alignment_net(eeg_embeddings, image_features)
            # print(f"  Alignment output shape: {alignment_outputs['aligned_features'].shape}")
            aligned_features = alignment_outputs['aligned_features']
            # Only debug first batch
            #if batch_idx == 0:
            #    break
            # Calculate losses
            # losses = self.criterion(aligned_features, image_features, routing_probs)
            losses = self.criterion(aligned_features, image_features, text_features, routing_probs) # update for clip text

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.alignment_net.parameters(),
                    self.config.gradient_clip_norm
                )

            self.optimizer.step()

            # Record losses
            for k, v in losses.items():
                epoch_losses[k].append(v.item())

            # Update progress bar
            # if batch_idx % 10 == 0:
            #     pbar.set_postfix({
            #         'loss': f"{losses['total']:.4f}",
            #         'mse': f"{losses['mse']:.4f}",
            #         'cos': f"{losses.get('cosine', 0):.3f}"
            #     })
            # Update progress bar to show both losses
            if batch_idx % 100 == 0:
                pbar.set_postfix({
                    'total': f"{losses['total']:.4f}",
                    'dino': f"{losses['contrastive_dino']:.4f}",
                    'clip': f"{losses.get('contrastive_clip', 0):.4f}",
                    'mse': f"{losses['mse']:.4f}"
                })

        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

        # Ensure required keys exist
        if 'total' not in avg_losses:
            avg_losses['total'] = avg_losses.get('mse', 0.0)
        if 'mse' not in avg_losses:
            avg_losses['mse'] = avg_losses.get('total', 0.0)

        return avg_losses

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.alignment_net.eval()

        epoch_losses = defaultdict(list)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val Epoch {self.current_epoch + 1}",
                        disable=(self.local_rank != 0))

            for batch in pbar:
                # Move to device
                eeg_embeddings = batch['eeg_embedding'].to(self.device)
                images = batch['image'].to(self.device)
                routing_probs = batch.get('routing_probs', None)
                if routing_probs is not None:
                    routing_probs = routing_probs.to(self.device)

                # Extract DinoV3 features
                dinov3_outputs = self.dinov3_encoder(images)
                image_features = dinov3_outputs['features']


                # Calculate losses
                # losses = self.criterion(aligned_features, image_features, routing_probs)
                # Should be:
                text_features = None
                if self.config.use_clip_text and 'text_caption' in batch:
                    text_features = self.dinov3_encoder.forward_text(batch['text_caption'])

                # Forward pass
                alignment_outputs = self.alignment_net(eeg_embeddings, image_features)
                aligned_features = alignment_outputs['aligned_features']

                losses = self.criterion(aligned_features, image_features, text_features, routing_probs)

                # Record losses
                for k, v in losses.items():
                    epoch_losses[k].append(v.item())

        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses

    def train(self,resume_from: Optional[str] = None):
        """Main training loop"""

        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info(f" Starting Phase 3 alignment training")
        logger.info(f"  Training on combined data from all subjects")
        logger.info(f"  Total training pairs: {len(self.train_dataset)}")
        logger.info(f"  Total validation pairs: {len(self.val_dataset)}")
        logger.info(f"  Training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Training
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)

            # Validation (every 5 epochs or first epoch)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_losses = self.validate_epoch()
                self.val_losses.append(val_losses)
            else:
                # Reuse last validation loss
                self.val_losses.append(self.val_losses[-1] if self.val_losses else train_losses)
                val_losses = self.val_losses[-1]

            if epoch >= 5:  # After warmup epochs
               self.scheduler.step(val_losses['total'])

            # ADD THIS - Save every single epoch unconditionally
            if self.local_rank == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
                logger.info(f"✓ Saved checkpoint for epoch {epoch + 1}")

            # Retrieval evaluation (every 10 epochs)
            # if (epoch + 1) % 10 == 0 and self.local_rank == 0:
            #     retrieval_results = evaluate_retrieval(
            #         self.alignment_net,
            #         self.dinov3_encoder,
            #         self.val_loader,
            #         self.device,
            #         self.config
            #     )
            #     logger.info(f"Retrieval metrics: {retrieval_results}")

            # Learning rate scheduling
            # NOTE: If using OneCycleLR, this is called per-batch in train_epoch()
            # If using ReduceLROnPlateau, uncomment below:
            # if epoch >= self.config.warmup_epochs:
            #     self.scheduler.step(val_losses['total'])

            # Logging (only on rank 0)
            if self.local_rank == 0:
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                logger.info(f"  Train - Total: {train_losses['total']:.4f}, "
                            f"DINO: {train_losses['contrastive_dino']:.4f}, "
                            f"CLIP: {train_losses.get('contrastive_clip', 0):.4f}, "
                            f"MSE: {train_losses['mse']:.4f}")
                logger.info(f"  Val   - Total: {val_losses['total']:.4f}")
                logger.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")

                # Save best model
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best_phase3_alignment_model.pth')
                    logger.info(f" New best model saved (val_loss: {self.best_val_loss:.4f})")

                # Save at epoch 10 for pipeline testing
                if epoch + 1 == 10:
                    self.save_checkpoint('checkpoint_epoch_10_TEST.pth')
                    logger.info(" Saved checkpoint at epoch 10 for pipeline testing!")

                # Save at milestone epochs
                if (epoch + 1) in [20, 30, 40, 50]:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

                # Regular checkpoint saving (every 10 epochs)
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f'dino_alignment_epoch_{epoch + 1}.pth')

        logger.info("Phase 3 alignment training completed!")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'config': self.config.__dict__,
            'alignment_net_state_dict': self.alignment_net.module.state_dict()
            if hasattr(self.alignment_net, 'module') else self.alignment_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f" Saved checkpoint: {filename}")

    def visualize_alignment(self):
        """Create comprehensive alignment visualizations"""
        logger.info(" Creating alignment visualizations...")

        self.alignment_net.eval()

        # Collect features for visualization
        n_samples = min(self.config.visualization_samples, len(self.val_dataset))
        indices = np.random.choice(len(self.val_dataset), n_samples, replace=False)

        eeg_features_list = []
        image_features_list = []
        aligned_features_list = []
        subject_ids_list = []
        categories_list = []
        routing_probs_list = []

        with torch.no_grad():
            for idx in tqdm(indices, desc="Collecting features"):
                sample = self.val_dataset[idx]

                # Process single sample
                eeg_embedding = sample['eeg_embedding'].unsqueeze(0).to(self.device)
                image = sample['image'].unsqueeze(0).to(self.device)

                # Get features
                dinov3_outputs = self.dinov3_encoder(image)
                image_features = dinov3_outputs['features']

                alignment_outputs = self.alignment_net(eeg_embedding, image_features)
                aligned_features = alignment_outputs['aligned_features']

                # Store
                eeg_features_list.append(eeg_embedding.cpu().numpy())
                image_features_list.append(image_features.cpu().numpy())
                aligned_features_list.append(aligned_features.cpu().numpy())
                subject_ids_list.append(sample['subject_id'])
                categories_list.append(sample['super_categories'])

                if 'routing_probs' in sample:
                    routing_probs_list.append(sample['routing_probs'].numpy())

        # Concatenate features
        #eeg_features = np.concatenate(eeg_features_list, axis=0)
        image_features = np.concatenate(image_features_list, axis=0)
        aligned_features = np.concatenate(aligned_features_list, axis=0)
        if routing_probs_list:
            routing_probs = np.stack(routing_probs_list, axis=0)
        else:
            routing_probs = None

        # Create visualizations
        #self._create_alignment_plots(eeg_features, image_features, aligned_features,
        #                             routing_probs, subject_ids_list, categories_list)


        # Save latent space visualization
        #self._visualize_latent_space(eeg_features, image_features, aligned_features,
        #                             subject_ids_list, categories_list)

    def _create_alignment_plots(self, eeg_features, image_features, aligned_features,
                                routing_probs, subject_ids, categories):
        """Create detailed alignment analysis plots"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Phase 3 Alignment Analysis - Epoch {self.current_epoch + 1}',
                     fontsize=16, fontweight='bold')

        # Plot 1: Feature magnitude comparison
        ax1 = axes[0, 0]
        #eeg_norms = np.linalg.norm(eeg_features, axis=1)
        image_norms = np.linalg.norm(image_features, axis=1)
        aligned_norms = np.linalg.norm(aligned_features, axis=1)

        #ax1.boxplot([eeg_norms, image_norms, aligned_norms],
        #            labels=['EEG', 'DinoV3', 'Aligned'])
        ax1.boxplot([image_norms, aligned_norms],
                    labels=['DinoV3', 'Aligned'])
        ax1.set_title('Feature Magnitude Distribution', fontweight='bold')
        ax1.set_ylabel('L2 Norm')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Alignment quality (cosine similarity)
        ax2 = axes[0, 1]
        similarities = []
        for i in range(len(aligned_features)):
            sim = cosine_similarity([aligned_features[i]], [image_features[i]])[0, 0]
            similarities.append(sim)

        ax2.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(similarities), color='red', linestyle='--',
                    label=f'Mean: {np.mean(similarities):.3f}')
        ax2.set_title('Alignment Quality (Cosine Similarity)', fontweight='bold')
        ax2.set_xlabel('Cosine Similarity')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Per-subject alignment quality
        ax3 = axes[0, 2]
        subject_similarities = defaultdict(list)
        for i, subj in enumerate(subject_ids):
            sim = cosine_similarity([aligned_features[i]], [image_features[i]])[0, 0]
            subject_similarities[subj].append(sim)

        subjects = list(subject_similarities.keys())[:10]  # Top 10 subjects
        subject_means = [np.mean(subject_similarities[s]) for s in subjects]

        ax3.bar(range(len(subjects)), subject_means)
        ax3.set_xticks(range(len(subjects)))
        ax3.set_xticklabels(subjects, rotation=45, ha='right')
        ax3.set_title('Per-Subject Alignment Quality', fontweight='bold')
        ax3.set_ylabel('Mean Cosine Similarity')
        ax3.grid(True, alpha=0.3)

     
        # Plot 5: Category-wise alignment
        ax5 = axes[1, 1]
        category_similarities = defaultdict(list)
        for i, cat in enumerate(categories):
            sim = cosine_similarity([aligned_features[i]], [image_features[i]])[0, 0]
            category_similarities[cat].append(sim)

        cats = list(category_similarities.keys())[:15]  # Top 15 categories
        cat_means = [np.mean(category_similarities[c]) for c in cats]

        ax5.barh(range(len(cats)), cat_means)
        ax5.set_yticks(range(len(cats)))
        ax5.set_yticklabels(cats, fontsize=8)
        ax5.set_title('Category-wise Alignment Quality', fontweight='bold')
        ax5.set_xlabel('Mean Cosine Similarity')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Feature correlation matrix
        ax6 = axes[1, 2]
        # Sample fewer features for visualization
        sample_idx = np.random.choice(len(aligned_features), 100, replace=True)
        corr_matrix = np.corrcoef(aligned_features[sample_idx], image_features[sample_idx])[:100, 100:]

        im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax6.set_title('Feature Correlation (Aligned vs DinoV3)', fontweight='bold')
        ax6.set_xlabel('DinoV3 Features')
        ax6.set_ylabel('Aligned EEG Features')
        plt.colorbar(im, ax=ax6, fraction=0.046)

        plt.tight_layout()
        save_path = self.output_dir / f'alignment_analysis_epoch_{self.current_epoch + 1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved alignment analysis plot")

    def _visualize_latent_space(self, eeg_features, image_features, aligned_features,
                                subject_ids, categories):
        """Visualize features in 2D latent space using t-SNE"""
        logger.info("Creating t-SNE visualization...")

        # Combine all features for joint t-SNE
        #all_features = np.vstack([eeg_features, image_features, aligned_features])
        all_features = np.vstack([image_features, aligned_features])  # Remove eeg_features

        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=self.config.tsne_perplexity,
                    n_iter=self.config.tsne_n_iter, random_state=42)
        features_2d = tsne.fit_transform(all_features)

        # Split back
        n_samples = len(image_features)
        image_2d = features_2d[:n_samples]
        aligned_2d = features_2d[n_samples:]

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Latent Space Visualization (t-SNE) - Epoch {self.current_epoch + 1}',
                     fontsize=14, fontweight='bold')

        # Plot 1: Feature types
        ax1 = axes[0]
        #ax1.scatter(eeg_2d[:, 0], eeg_2d[:, 1], alpha=0.5, s=10, label='EEG', c='blue')
        ax1.scatter(image_2d[:, 0], image_2d[:, 1], alpha=0.5, s=10, label='DinoV3', c='red')
        ax1.scatter(aligned_2d[:, 0], aligned_2d[:, 1], alpha=0.5, s=10, label='Aligned', c='green')
        ax1.set_title('Feature Space Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Alignment trajectories
        ax2 = axes[1]
        for i in range(0, n_samples, 10):  # Sample for clarity
            ax2.arrow(eeg_2d[i, 0], eeg_2d[i, 1],
                      aligned_2d[i, 0] - eeg_2d[i, 0],
                      aligned_2d[i, 1] - eeg_2d[i, 1],
                      alpha=0.3, width=0.1, head_width=0.5, color='blue')
            ax2.arrow(aligned_2d[i, 0], aligned_2d[i, 1],
                      image_2d[i, 0] - aligned_2d[i, 0],
                      image_2d[i, 1] - aligned_2d[i, 1],
                      alpha=0.3, width=0.1, head_width=0.5, color='green')
        ax2.set_title('Alignment Trajectories (EEG → Aligned → DinoV3)')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Category clustering
        ax3 = axes[2]
        unique_categories = list(set(categories))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))

        for cat_idx, cat in enumerate(unique_categories[:20]):  # Limit to 20 categories
            cat_mask = [c == cat for c in categories]
            cat_aligned = aligned_2d[cat_mask]
            if len(cat_aligned) > 0:
                ax3.scatter(cat_aligned[:, 0], cat_aligned[:, 1],
                            c=[colors[cat_idx]], s=20, alpha=0.6, label=cat)

        ax3.set_title('Category Clustering in Aligned Space')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / f'latent_space_epoch_{self.current_epoch + 1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved latent space visualization")

    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        if not self.train_losses:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Progress - Epoch {self.current_epoch + 1}', fontsize=14, fontweight='bold')

        epochs = range(1, len(self.train_losses) + 1)

        # Plot different loss components
        loss_keys = ['total', 'mse', 'cosine', 'contrastive', 'triplet', 'perceptual']

        for idx, key in enumerate(loss_keys):
            ax = axes[idx // 3, idx % 3]

            train_vals = [losses.get(key, 0) for losses in self.train_losses]
            val_vals = [losses.get(key, 0) for losses in self.val_losses]

            if any(v > 0 for v in train_vals):
                ax.plot(epochs, train_vals, label='Train', marker='o', markersize=3)
                ax.plot(epochs, val_vals, label='Val', marker='s', markersize=3)
                ax.set_title(f'{key.capitalize()} Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No {key} loss', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(f'{key.capitalize()} Loss')

        plt.tight_layout()
        save_path = self.output_dir / 'training_curves.png'
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def parse_arguments():
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description='Phase 3: EEG-DinoV3 Alignment Training'
        )

        # Data paths
        # parser.add_argument('--eeg_model_path', type=str, required=True,
        #                     help='Path to trained EEGViTCNet model checkpoint')
        # parser.add_argument('--eeg_data_dir', type=str, required=True,
        #                     help='Path to harmonized EEG data directory')
        # parser.add_argument('--images_dir', type=str, required=True,
        #                     help='Path to images directory')
        # parser.add_argument('--output_dir', type=str, required=True,
        #                     help='Output directory for results')

        # Replace eeg_model_path and eeg_data_dir with extracted_embeddings_dir
        parser.add_argument('--extracted_embeddings_dir', type=str, required=True,
                            help='Path to extracted EEG embeddings directory')
        parser.add_argument('--images_dir', type=str, required=True,
                            help='Path to images directory')
        parser.add_argument('--output_dir', type=str, required=True,
                            help='Output directory for results')

        parser.add_argument('--dinov3_model', type=str, default='dinov3_vitb16',
                            choices=['dinov3_vitb14', 'dinov3_vitb16', 'dinov3_vitl14', 'dinov3_vitg14'],
                            help='DinoV3 model variant')
        parser.add_argument('--use_cross_attention', action='store_true', default=True,
                            help='Use cross-modal attention')
        parser.add_argument('--use_contrastive', action='store_true', default=True,
                            help='Use contrastive learning')

        # Training parameters
        parser.add_argument('--batch_size', type=int, default=24,
                            help='Batch size')
        parser.add_argument('--learning_rate', type=float, default=5e-5,
                            help='Learning rate')
        parser.add_argument('--num_epochs', type=int, default=150,
                            help='Number of training epochs')
        parser.add_argument('--warmup_epochs', type=int, default=15,
                            help='Number of warmup epochs')

        # Distributed training
        parser.add_argument('--use_distributed', action='store_true', default=True,
                            help='Use distributed training')
        parser.add_argument('--local_rank', type=int, default=0,
                            help='Local rank for distributed training')

        parser.add_argument('--use_clip_text', action='store_true',
                            help='Enable CLIP text alignment in addition to DINOv3')
        parser.add_argument('--clip_weight', type=float, default=0.3,
                            help='Weight for CLIP text loss')
        parser.add_argument('--holdout_ratio', type=float, default=0.2,
                            help='Ratio of images to hold out for retrieval testing')
        # Visualization
        parser.add_argument('--visualization_samples', type=int, default=2500,
                            help='Number of samples for visualization')
        parser.add_argument('--resume_from', type=str, default=None,
                            help='Path to checkpoint to resume from')

        return parser.parse_args()

def main():
        """Main training function"""
        args = parse_arguments()

        # Create configuration
        config = Phase3AlignmentConfig(
            dinov3_model_name=args.dinov3_model,
            use_cross_attention=args.use_cross_attention,
            use_contrastive_learning=args.use_contrastive,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            warmup_epochs=args.warmup_epochs,
            use_distributed=args.use_distributed,
            visualization_samples=args.visualization_samples,
            use_clip_text=args.use_clip_text,
            clip_weight=args.clip_weight,
            holdout_ratio=args.holdout_ratio

        )

        logger.info(" Starting Phase 3: EEG-DinoV3 Alignment Training")
        logger.info(f"Configuration: {config}")

        # Create trainer
        # trainer = Phase3AlignmentTrainer(
        #     config=config,
        #     eeg_model_path=args.eeg_model_path,
        #     eeg_data_dir=args.eeg_data_dir,
        #     images_dir=args.images_dir,
        #     output_dir=args.output_dir
        # )

        trainer = Phase3AlignmentTrainer(
            config=config,
            extracted_embeddings_dir=args.extracted_embeddings_dir,  # Changed
            images_dir=args.images_dir,  # Changed
            output_dir=args.output_dir
        )

        # resume_checkpoint = output_dir / 'best_phase3_alignment_model.pth'
        # if resume_checkpoint.exists():
        #     logger.info(f"Resuming from {resume_checkpoint}")
        #     trainer.load_checkpoint(str(resume_checkpoint))
        if args.resume_from:
            trainer.train(resume_from=args.resume_from)
        else:
            trainer.train()
        # Start training
        # trainer.train()

        logger.info(" Phase 3 training completed successfully!")

if __name__ == "__main__":
        main()