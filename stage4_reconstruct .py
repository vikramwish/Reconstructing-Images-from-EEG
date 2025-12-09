import os
import warnings
import sys

import os
os.environ["CUDA_NVML_DISABLED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Add this
os.environ["NCCL_NVML_ENABLED"] = "0"  # Add this

# Keeping my existing ones
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


os.environ["NCCL_SOCKET_IFNAME"] = "lo"  # Force localhost interface
os.environ["NCCL_IB_DISABLE"] = "1"      # Disable InfiniBand
os.environ["NCCL_DEBUG"] = "WARN"        # Less verbose
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Use new env var name

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*weights_only.*')
os.environ['NUMEXPR_MAX_THREADS'] = '8'

sys.path.append('/raid/datasets/tanaya/fm/script_f/new_vd/versatile-diffusion')

from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.model_zoo.vd import VD

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import h5py
import csv
import logging
import numpy as np
import argparse
import time
from datetime import timedelta
import math
from torch.utils.data import random_split
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.models import alexnet, efficientnet_b0, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor
# from torchmetrics.image import InceptionScore
import clip

# LOGGING
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# EEG → Image Embedding Dataset
# ============================================================================
class EEGEmbeddingDataset(Dataset):
    def __init__(self, h5_path, image_dir, max_samples=None):
        self.h5_path = h5_path
        self.image_dir = image_dir

        with h5py.File(h5_path, "r") as f:
            self.length = len(f["aligned_embeddings"])
            self.image_paths = [
                p.decode() if isinstance(p, bytes) else str(p)
                for p in f["image_paths"][:]
            ]

        if max_samples:
            self.length = min(self.length, max_samples)
            self.image_paths = self.image_paths[:max_samples]

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f"Dataset: {self.length} samples from {image_dir}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # real_idx = (idx + np.random.randint(0, 100)) % self.length
        with h5py.File(self.h5_path, "r") as f:
            emb = torch.tensor(f["aligned_embeddings"][idx]).float()
            # emb = torch.tensor(f["aligned_embeddings"][real_idx]).float()
            # emb = (emb - emb.mean()) / (emb.std() + 1e-8)  # noramlize
            emb = F.normalize(emb, p=2, dim=-1)  # L2

        img_filename = self.image_paths[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.warning(f"Failed to load {img_path}: {e}")
            img = torch.zeros(3, 512, 512)

        return emb, img


# ============================================================================
# Extended Visual Metrics
# ============================================================================
class ExtendedMetrics:
    def __init__(self, device="cuda"):
        self.device = device
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
        alex = alexnet(weights="IMAGENET1K_V1").to(device).eval()
        self.alex2 = create_feature_extractor(alex, return_nodes={"features.2": "layer2"})
        self.alex5 = create_feature_extractor(alex, return_nodes={"features.5": "layer5"})
        eff = efficientnet_b0(weights="IMAGENET1K_V1").to(device).eval()
        self.effnet = create_feature_extractor(eff, return_nodes={"avgpool": "embed"})
        self.inception = inception_v3(weights="IMAGENET1K_V1").to(device).eval()
        self.preproc = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize([0.5], [0.5])
        ])

    @torch.no_grad()
    def inception_score(self, imgs, splits=1):
        imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
        preds = F.softmax(self.inception(imgs), dim=1).cpu().numpy()
        scores, N = [], len(preds)
        for k in range(splits):
            part = preds[k * (N // splits):(k + 1) * (N // splits)]
            py = np.mean(part, axis=0)
            scores.append(np.exp(np.mean([np.sum(pyx * np.log(pyx / (py + 1e-10))) for pyx in part])))
        return float(np.mean(scores))

    @torch.no_grad()
    def evaluate(self, gen, tgt):
        gen, tgt = gen.to(self.device), tgt.to(self.device)
        out = {}
        out["pixcorr"] = F.cosine_similarity(gen.flatten(1), tgt.flatten(1), dim=1).mean().item()
        out["ssim"] = self.ssim(gen, tgt).item()
        out["lpips"] = self.lpips(gen, tgt).item()
        out["mse"] = F.mse_loss(gen, tgt).item()
        out["inception_score"] = self.inception_score(gen)
        feat_a2g = self.alex2(self.preproc(gen))["layer2"].flatten(1)
        feat_a2t = self.alex2(self.preproc(tgt))["layer2"].flatten(1)
        feat_a5g = self.alex5(self.preproc(gen))["layer5"].flatten(1)
        feat_a5t = self.alex5(self.preproc(tgt))["layer5"].flatten(1)
        feg = self.effnet(self.preproc(gen))["embed"].flatten(1)
        fet = self.effnet(self.preproc(tgt))["embed"].flatten(1)
        out["alexnet_2"] = F.cosine_similarity(feat_a2g, feat_a2t, dim=1).mean().item()
        out["alexnet_5"] = F.cosine_similarity(feat_a5g, feat_a5t, dim=1).mean().item()
        out["efficientnet"] = F.cosine_similarity(feg, fet, dim=1).mean().item()
        return out



import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGToVDAdapter(nn.Module):

    def __init__(self, eeg_dim=768, spatial_size=16):
        super().__init__()
        self.eeg_dim = eeg_dim
        self.spatial_size = spatial_size
        target_dim = 4 * spatial_size * spatial_size  # AutoKL latent shape

        # ----- EEG → image latent -----
        self.to_image = nn.Sequential(
            nn.Linear(self.eeg_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(2048, target_dim)
        )

        # ----- EEG → CLIP vision embedding -----
        self.to_clip_vision = nn.Sequential(
            nn.Linear(self.eeg_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512)
        )

        # ----- EEG → CLIP text embedding -----
        self.to_clip_text = nn.Sequential(
            nn.Linear(self.eeg_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512)
        )

        # Normalization and adaptive scaling
        self.input_norm = nn.LayerNorm(self.eeg_dim)
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.var_gain_scale = nn.Parameter(torch.tensor(5.0))
        self.var_gain_bias = nn.Parameter(torch.tensor(0.0))

        # initialize weights
        self._init_weights()

    # -------------------------------------------------------
    # Weight initialization
    # -------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.2)  # slightly stronger init
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------
    def forward(self, eeg_emb):
        """
        Args:
            eeg_emb: (B, eeg_dim) EEG-aligned embedding
        Returns:
            img:        (B, 4, H, W) AutoKL latent
            clip_v:     (B, 512) CLIP vision embedding
            clip_t:     (B, 512) CLIP text embedding (proxy)
        """
        B = eeg_emb.size(0)
        eeg_emb = self.input_norm(eeg_emb)

        # --- spatial latent projection ---
        img = self.to_image(eeg_emb)
        img = img.view(B, 4, self.spatial_size, self.spatial_size)

        # --- center only, keep amplitude info ---
        mean = img.mean(dim=(1, 2, 3), keepdim=True)
        img = img - mean

        # --- global amplitude scaling ---
        img = img * torch.exp(self.log_temp)

        # --- differentiable variance gain ---
        var = img.var(dim=(1, 2, 3), keepdim=True)
        target_var = 0.05
        gain = torch.sigmoid(self.var_gain_scale * (var - target_var) + self.var_gain_bias) + 0.5
        img = img * gain

        # --- mild tanh scaling to match VAE latent range ---
        img = 0.18215 * torch.tanh(img / 2.0)

        # --- CLIP branches ---
        clip_v = F.normalize(self.to_clip_vision(eeg_emb), p=2, dim=-1)
        clip_t = F.normalize(self.to_clip_text(eeg_emb), p=2, dim=-1)

        return img, clip_v, clip_t

    # -------------------------------------------------------
    # Optional decorrelation regularizer
    # -------------------------------------------------------
    def decorrelation_loss(self, feats):
        B, D = feats.shape
        c = (feats - feats.mean(0)).T @ (feats - feats.mean(0)) / (B - 1)
        off_diag = c - torch.diag(torch.diag(c))
        return (off_diag ** 2).sum() / D
# ============================================================================
# DDP Utilities
# ============================================================================
def setup(rank, world_size):
    """Initialize DDP with better error handling"""
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"  # Changed from "localhost"
        # os.environ["MASTER_PORT"] = "12355"
        # os.environ["RANK"] = str(rank)
        # os.environ["LOCAL_RANK"] = str(rank)
        # os.environ["WORLD_SIZE"] = str(world_size)

        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            init_method='env://'
        )
        torch.cuda.set_device(rank)

        if rank == 0:
            logger.info(f"DDP initialized with {world_size} processes")

        # Simpler communication test
        dist.barrier()

        if rank == 0:
            logger.info(" DDP communication test passed")

    except Exception as e:
        logger.error(f"[GPU {rank}] DDP setup failed: {e}")
        raise


def cleanup():
    """Cleanup DDP with error handling"""
    try:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_default_dtype(torch.float32)


def train_ddp(rank, world_size, args):
    """
    Distributed EEG → Versatile Diffusion training loop.
    Trains the adapter + unfreezes VD attention layers for fine-tuning.
    """
    setup_success = False
    try:
        # Initialize DDP
        setup(rank, world_size)
        setup_success = True
        device = torch.device(f"cuda:{rank}")
        #
        # # Load AFTER DDP init - can be broadcast from rank 0
        # if rank == 0:
        #     logger.info("Loading pretrained VD model...")
        #
        # # Load pretrained VD backbone
        # # logger.info(f"[GPU {rank}] Loading pretrained VD model...")
        # cfgm_name = 'vd_noema'
        # cfgm = model_cfg_bank()(cfgm_name)
        # net = get_model()(cfgm)
        # sd = torch.load(args.pretrained_vd_path, map_location='cpu')
        # net.load_state_dict(sd, strict=False)
        #
        # net.to(device)
        # net.device = device
        #
        # # Convert entire model to half precision
        # # net = net.half()
        # net = net

        # ============================================
        # FIXED: Load model on ALL ranks
        # ============================================
        if rank == 0:
            logger.info("Loading pretrained VD model...")

        try:
            cfgm_name = 'vd_noema'
            cfgm = model_cfg_bank()(cfgm_name)
            net = get_model()(cfgm)
            sd = torch.load(args.pretrained_vd_path, map_location='cpu')
            net.load_state_dict(sd, strict=False)

            # Check if model loaded
            if net is None:
                raise RuntimeError(f"[GPU {rank}] get_model() returned None!")

            # Load weights on CPU first (all ranks do this)
            sd = torch.load(args.pretrained_vd_path, map_location='cpu')
            net.load_state_dict(sd, strict=False)

            if rank == 0:
                logger.info(f"✅ Model loaded successfully: {type(net)}")

        except Exception as e:
            logger.error(f"[GPU {rank}] Failed to load model: {e}")
            raise

        # ============================================
        # FIX: Don't reassign net.to() result
        # ============================================
        if rank == 0:
            logger.info(f"Moving model to {device}...")

        # Move to device
        # net = net.to(device)
        # net.device = device
        net.to(device)

        # Manually set device attribute
        net.device = device

        if rank == 0:
            logger.info(f" Model moved to {device}, type: {type(net)}")

        # OPTION 1: Freeze VD (Recommended)
        # ============================================
        for p in net.parameters():
            p.requires_grad = False

        net.eval()  # Keep frozen model in eval mode

        if rank == 0:
            logger.info(" VD model frozen (eval mode)")


        if hasattr(net, 'logvar'):
            net.logvar = net.logvar.to(device)

        # Explicit diffusion model device
        if hasattr(net, 'model') and hasattr(net.model, 'diffusion_model'):
            net.model.diffusion_model.device = device

        # Assign AutoKL (latent encoder-decoder) - keep in half
        if hasattr(net, "autokl"):
            if hasattr(net.autokl, "encoder"):
                net.autokl.encoder = net.autokl.encoder.to(device)
            if hasattr(net.autokl, "decoder"):
                net.autokl.decoder = net.autokl.decoder.to(device)
            if rank == 0:
                logger.info(f" AutoKL moved to {device}")

            # Initialize adapter
        adapter = EEGToVDAdapter(
            eeg_dim=args.eeg_dim,
            spatial_size=args.spatial_size
        ).to(device)

        if rank == 0:
            logger.info(f"Adapter initialized on {device}")

        # ============================================
        # Load External CLIP Model
        # ============================================
        if rank == 0:
            logger.info("Loading external CLIP model...")

        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        # clip_model = clip_model.half()  #  Convert to fp16
        clip_model.eval()

        # Freeze CLIP
        for param in clip_model.parameters():
            param.requires_grad = False

        if rank == 0:
            logger.info(" CLIP ViT-B/32 loaded and frozen")

   
        # Wait for all ranks to load
        dist.barrier()

        # ========================================
        # ADD PERCEPTUAL LOSS MODEL
        # ========================================
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device).eval()

        if rank == 0:
            logger.info(" LPIPS perceptual loss initialized")

        # ============================================
        # DON'T wrap frozen net in DDP
        # Only wrap trainable adapter
        # ============================================
        # adapter = DDP(
        #     adapter,
        #     device_ids=[rank],
        #     output_device=rank,
        #     find_unused_parameters=True
        #
        # )

        if rank == 0:
            logger.info(" Adapter wrapped in DDP")

        # Initialize EEG → VD Adapter
        # adapter = EEGToVDAdapter(eeg_dim=args.eeg_dim, spatial_size=args.spatial_size).to(device).half()

        # for p in adapter.parameters():
        #     p.requires_grad = False
        #
        # net = net.to(device)
        # net.eval()  # Keep in eval mode

        # Wrap both in DDP
        # net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=True, broadcast_buffers=True)
        # adapter = DDP(adapter, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        # if any(p.requires_grad for p in adapter.parameters()):
        #     adapter = DDP(adapter, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        # else:
        #     print(f"[GPU {rank}] Adapter has no trainable parameters — skipping DDP.")
        # print(f"[GPU {rank}] Running in single-GPU inference mode.")

        # Dataset & Dataloader
        dataset = EEGEmbeddingDataset(args.embeddings_path, image_dir=args.image_dir, max_samples=args.max_samples)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                           drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        effective_batch_size = max(1, args.batch_size // world_size)

        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, sampler=train_sampler,
                                  num_workers=8, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, sampler=val_sampler,
                                num_workers=8, pin_memory=True, drop_last=True)

        if rank == 0:
            logger.info(f"Training samples: {len(train_dataset)} | Validation: {len(val_dataset)}")
            logger.info(f"Batch per GPU: {effective_batch_size} | Total batch: {args.batch_size}")



        # Optimizer, Scheduler, GradScaler
        # optimizer = torch.optim.AdamW(
        #     list(filter(lambda p: p.requires_grad, net.parameters())) +
        #     list(adapter.parameters()),
        #     lr=args.learning_rate,
        #     weight_decay=0.01
        # )

        optimizer = torch.optim.AdamW(
            list(adapter.parameters()),  #  Only adapter
            lr=args.learning_rate,
            weight_decay=0.05    #changed from 0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        # # scaler = torch.cuda.amp.GradScaler()
        #

   

        resume_ckpt = "/tmp/tanaya/diffusion_res2/checkpoints/checkpoint_epoch_004.pth"  # your last saved checkpoint

        if os.path.exists(resume_ckpt):
            print(f"Resuming training from {resume_ckpt}")
            ckpt = torch.load(resume_ckpt, map_location=device)

            # --- Adapter weights ---
            if "adapter" in ckpt:
                sd = ckpt["adapter"]
                adapter.load_state_dict(sd, strict=False)

 
                with torch.no_grad():
                    adapter.log_temp = nn.Parameter(sd["log_temp"].clone())
                    adapter.var_gain_scale = nn.Parameter(sd["var_gain_scale"].clone())
                    adapter.var_gain_bias = nn.Parameter(sd["var_gain_bias"].clone())
                print("🔧 Restored log_temp, var_gain_scale, var_gain_bias manually.")

            else:
                print(" No 'adapter' key found in checkpoint — loading entire state_dict.")
                adapter.load_state_dict(ckpt, strict=False)

            # --- Optimizer and scheduler ---
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    print(" Optimizer state restored.")
                except Exception as e:
                    print(f"Optimizer state load failed ({e}); reinitializing optimizer.")
            else:
                print(" No optimizer state found in checkpoint.")

            if "scheduler" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                    print("Scheduler state restored.")
                except Exception as e:
                    print(f" Scheduler state load failed ({e}); reinitializing scheduler.")
            else:
                print(" No scheduler state found in checkpoint.")

            # --- Epoch and LR setup ---
            start_epoch = ckpt.get("epoch", 4) + 1
            for g in optimizer.param_groups:
                g["lr"] = 5e-5

            print(f" Resume complete. Starting from epoch {start_epoch} with lr={optimizer.param_groups[0]['lr']}")
        else:
            print("No checkpoint found — starting from scratch.")
            start_epoch = 0

        adapter = DDP(
            adapter,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False

        )

        metrics = ExtendedMetrics(device=device) if rank == 0 else None

        # CSV and checkpoint setup
        if rank == 0:
            checkpoint_dir = Path(args.output_dir) / "checkpoints"
            generated_dir = Path(args.output_dir) / "generated"
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(generated_dir, exist_ok=True)

            csv_path = Path(args.output_dir) / "training_metrics.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "pixcorr", "ssim", "lpips",
                                 "alexnet_2", "alexnet_5", "efficientnet", "inception_score",
                                 "composite_score", "learning_rate", "epoch_time_sec"])

        best_comp = 0.0
        dist.barrier()


        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cuda.max_split_size_mb = 512

         # AFTER ADAPTER INITIALIZATION, BEFORE TRAINING LOOP


        # print("log_temp:", adapter.log_temp.item(),
        #       "scale:", adapter.var_gain_scale.item(),
        #       "bias:", adapter.var_gain_bias.item())
        if hasattr(adapter, "module"):
            print("log_temp:", adapter.module.log_temp.item(),
                  "scale:", adapter.module.var_gain_scale.item(),
                  "bias:", adapter.module.var_gain_bias.item())
        else:
            print("log_temp:", adapter.log_temp.item(),
                  "scale:", adapter.var_gain_scale.item(),
                  "bias:", adapter.var_gain_bias.item())

        # Training Loop
        # for epoch in range(args.epochs):
        for epoch in range(start_epoch, args.epochs):

            epoch_start = time.time()
            train_sampler.set_epoch(epoch)
            net.train()
            adapter.train()

            total_loss = 0.0
            num_batches = 0

            if rank == 0:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            else:
                pbar = train_loader

            # # Training loop with gradient accumulation
            accumulation_steps = 4 #changed from 16 to 8


                 for batch_idx, (eeg_emb, target_img) in enumerate(pbar):
                eeg_emb = eeg_emb.to(device, non_blocking=True).float()
                target_img = target_img.to(device, non_blocking=True).float()

                # ========================================
                # Get Target Embeddings
                # ========================================
                with torch.no_grad():
                    # 1. Target latent (spatial)
                    target_latent = net.autokl_encode(target_img)

                    # 2. Target CLIP vision (semantic)
                    # Denormalize VD's [-1,1] to [0,1]
                    target_img_01 = (target_img + 1) / 2

                    # Resize to CLIP's 224x224
                    target_img_224 = F.interpolate(target_img_01, size=(224, 224), mode='bilinear')

                    # Normalize for CLIP (ImageNet stats)
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
                    target_img_clip = (target_img_224 - mean) / std

                    # Encode with CLIP
                    target_clip_vision = clip_model.encode_image(target_img_clip).float()
                    # target_clip_vision = clip_model.encode_image(target_img_clip.half()).float()
                    target_clip_vision = F.normalize(target_clip_vision, p=2, dim=-1)

                    # 3. Target CLIP text (use vision as proxy)
                    target_clip_text = target_clip_vision

                # ========================================
                # Adapter Forward
                # ========================================
                img_emb, pred_clip_vision, pred_clip_text = adapter(eeg_emb)


                # Resize latent if needed
                if img_emb.shape != target_latent.shape:
                    img_emb_resized = F.interpolate(
                        img_emb,
                        size=target_latent.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    img_emb_resized = img_emb

                # ========================================
                # Multi-Task Loss
                # ========================================

                # 1. Latent Loss (Spatial)
                latent_loss = F.mse_loss(img_emb_resized, target_latent)

                # 2. CLIP Vision Loss (Semantic)
                clip_vision_loss = F.mse_loss(pred_clip_vision, target_clip_vision)

                # 3. CLIP Text Loss (Semantic)
                clip_text_loss = F.mse_loss(pred_clip_text, target_clip_text)



                # --- Normalize embeddings for contrastive ----
                pred_clip_vision = F.normalize(pred_clip_vision, dim=-1)
                pred_clip_text = F.normalize(pred_clip_text, dim=-1)
                target_clip_vision = F.normalize(target_clip_vision, dim=-1)

                # Contrastive InfoNCE loss ----
                temperature = 0.07
                logits_per_eeg = pred_clip_vision @ target_clip_vision.T / temperature
                logits_per_target = target_clip_vision @ pred_clip_vision.T / temperature
                labels = torch.arange(logits_per_eeg.size(0), device=device)

                contrastive_loss = (
                                           F.cross_entropy(logits_per_eeg, labels) +
                                           F.cross_entropy(logits_per_target, labels)
                                   ) / 2

                # Mild variance regularization ----
                clip_var = pred_clip_vision.var(dim=0).mean()
                var_loss = 0.01 * torch.exp(-10 * clip_var)

                # Add debugging every 50 batches
                if rank == 0 and batch_idx % 50 == 0:
                    with torch.no_grad():
                        # Check if predictions are diverse
                        pred_std = pred_clip_vision.std(dim=0).mean().item()
                        target_std = target_clip_vision.std(dim=0).mean().item()

                        # Check if all predictions are the same
                        pairwise_dist = torch.cdist(pred_clip_vision, pred_clip_vision, p=2).mean().item()

                        logger.info(
                            f"[Batch {batch_idx}] "
                            f"Pred std: {pred_std:.4f}, Target std: {target_std:.4f}, "
                            f"Pairwise dist: {pairwise_dist:.4f}"
                        )

                        # If pred_std < 0.01, predictions are collapsed!
                        if pred_std < 0.01:
                            logger.error(" CLIP predictions collapsed to constant!")

                # 4. Perceptual Loss (Optional)
                if batch_idx % 2 == 0:
                    net.autokl.decoder.eval()
                    gen_imgs = net.autokl_decode(img_emb_resized)

                    with torch.no_grad():
                        target_imgs = net.autokl_decode(target_latent)

                    if torch.isnan(gen_imgs).any():
                        perceptual_loss = torch.tensor(0.0, device=device)
                    else:
                        gen_imgs_norm = torch.clamp((gen_imgs + 1) / 2, 0, 1)
                        target_imgs_norm = torch.clamp((target_imgs + 1) / 2, 0, 1)
                        perceptual_loss = lpips_model(gen_imgs_norm, target_imgs_norm)
                else:
                    perceptual_loss = torch.tensor(0.0, device=device)


                clip_weight = 1.0  # emphasize semantic consistency after resuming
                loss = (
                        1.0 * latent_loss +
                        clip_weight * clip_vision_loss +
                        0.5 * clip_text_loss +
                        0.2 * perceptual_loss+ 0.2 * contrastive_loss + var_loss
                )

                loss = loss / accumulation_steps
                loss.backward()

                # ========================================
                # Optimizer Step
                # ========================================
                if (batch_idx + 1) % accumulation_steps == 0:
                    total_norm = 0.0
                    for p in adapter.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5

                    if rank == 0 and batch_idx % 20 == 0:
                        logger.info(f"[Batch {batch_idx}] Grad norm: {total_norm:.6f}")

                    grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()  # ← Add this

                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()

                total_loss += loss.item()
                num_batches += 1

                if rank == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lat=f"{latent_loss.item():.4f}",
                        cv=f"{clip_vision_loss.item():.4f}",  # ← Monitor CLIP loss
                        ct=f"{clip_text_loss.item():.4f}",
                        per=f"{perceptual_loss.item():.4f}"
                    )

                # ========================================
                # Monitoring
                # ========================================
                if rank == 0 and batch_idx % 10 == 0:
                    with torch.no_grad():
                        latent_var = img_emb_resized.var().item()
                        latent_mean = img_emb_resized.mean().item()

                        #  Also monitor CLIP prediction quality
                        clip_sim = F.cosine_similarity(pred_clip_vision, target_clip_vision, dim=1).mean().item()

                        logger.info(
                            f"[Batch {batch_idx}] "
                            f"Latent var: {latent_var:.6f}, mean: {latent_mean:.6f}, "
                            f"CLIP sim: {clip_sim:.4f}"
                        )

                        # # Check diversity
                        # var_across_samples = out1
                        # # logger.info(f"[diversity: {var_across_samples}]")

                # Save samples

                if rank == 0 and batch_idx % 50 == 0:
                    with torch.no_grad():
                        # Access the underlying module if wrapped in DDP
                        adapter_mod = adapter.module if hasattr(adapter, "module") else adapter

                        # Sample variance of image latents
                        img_var = img_emb_resized.var().item()

                        # Global temperature
                        log_temp = adapter_mod.log_temp.item()

                        # CLIP projection std
                        clip_std = pred_clip_vision.std(dim=0).mean().item()

                        logger.info(
                            f"[Adapter Monitor] log_temp={log_temp:.3f}, "
                            f"latent_var={img_var:.4f}, clip_std={clip_std:.4f}"
                        )

                        if img_var < 0.02:
                            logger.warning(" Latent variance too low — watch for collapse.")
                        if img_var > 0.15:
                            logger.warning(" Latent variance high — may cause instability.")

                if rank == 0 and batch_idx % 200 == 0:
                    with torch.no_grad():
                        gen_latent = img_emb_resized[:1]

                        gen_img_scaled = net.autokl_decode(gen_latent * 1.5)
                        gen_img = net.autokl_decode(gen_latent)

                        if not torch.isnan(gen_img).any():
                            save_image(
                                torch.clamp((gen_img + 1) / 2, 0, 1),
                                generated_dir / f"train_epoch_{epoch + 1:03d}_batch_{batch_idx:04d}.png"
                            )
                            save_image(
                                torch.clamp((gen_img_scaled + 1) / 2, 0, 1),
                                generated_dir / f"train_epoch_{epoch + 1:03d}_batch_{batch_idx:04d}_scaled.png"
                            )

            scheduler.step()
            avg_train_loss = total_loss / max(num_batches, 1)
            loss_tensor = torch.tensor(avg_train_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_train_loss = loss_tensor.item()
            epoch_time = time.time() - epoch_start
            lr_now = optimizer.param_groups[0]["lr"]



            # validation loop

            if rank == 0:
                net.eval()
                adapter.eval()

                with torch.no_grad():
                    # ========================================
                    # Validation Loss (same as training loss)
                    # ========================================
                    val_loss = 0.0
                    val_count = 0
                    for val_idx, (eeg_emb, target_img) in enumerate(val_loader):
                        eeg_emb = eeg_emb.to(device).float()
                        target_img = target_img.to(device).float()

                        # Encode target
                        target_latent = net.autokl_encode(target_img)

                        # Adapter forward
                        # img_emb, text_cond = adapter(eeg_emb)
                        # Adapter forward
                        img_emb, pred_clip_vision, pred_clip_text = adapter(eeg_emb)

                        # Resize if needed
                        if img_emb.shape != target_latent.shape:
                            img_emb_resized = F.interpolate(
                                img_emb,
                                size=target_latent.shape[2:],
                                mode='bilinear',
                                align_corners=False
                            )
                        else:
                            img_emb_resized = img_emb

                        # Same loss as training
                        latent_loss = F.mse_loss(img_emb_resized, target_latent)

                        loss = latent_loss

                        val_loss += loss.item()
                        val_count += 1
                        if val_idx >= 100:
                            break

                    avg_val_loss = val_loss / max(1, val_count)

                    # ========================================
                    # Generate Samples (DIRECT DECODE ONLY)
                    # ========================================
                    sample_emb, sample_img = next(iter(val_loader))
                    sample_emb = sample_emb[:4].to(device).float()
                    sample_img = sample_img[:4].to(device).float()

                    # Adapter forward
                    coarse_latent, pred_clip_vision, pred_clip_text = adapter(sample_emb)

                    # Resize to 64x64 if needed
                    if coarse_latent.shape[2] != 64:
                        coarse_latent = F.interpolate(
                            coarse_latent,
                            size=(64, 64),
                            mode='bilinear'
                        )

                    # DIRECT DECODE (skip diffusion for now)
                    # gen_imgs = net.autokl_decode(coarse_latent)
                    gen_imgs = net.autokl_decode(coarse_latent * 1.5)

                    gen_imgs = torch.clamp((gen_imgs + 1) / 2, 0, 1)
                    real_imgs = torch.clamp((sample_img + 1) / 2, 0, 1)

                    # Save comparison
                    comparison = torch.cat([real_imgs, gen_imgs], dim=0)
                    save_image(
                        comparison,
                        generated_dir / f"val_epoch_{epoch + 1:03d}_comparison.png",
                        nrow=4
                    )

                    #  Compute metrics on direct decode
                    if metrics is not None:
                        gen_resized = F.interpolate(gen_imgs, (224, 224))
                        real_resized = F.interpolate(real_imgs, (224, 224))
                        m = metrics.evaluate(gen_resized, real_resized)
                        composite = (0.35 * m["pixcorr"] + 0.25 * m["ssim"] +
                                     0.20 * (1 - m["lpips"]) + 0.20 * m["inception_score"] / 10)
                        m["composite_score"] = composite
                    else:
                        m = {"composite_score": 0.0}

                    # Write CSV
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch + 1, avg_train_loss, avg_val_loss,
                                         m.get("pixcorr", 0), m.get("ssim", 0), m.get("lpips", 0),
                                         m.get("alexnet_2", 0), m.get("alexnet_5", 0),
                                         m.get("efficientnet", 0), m.get("inception_score", 0),
                                         composite, lr_now, epoch_time])
                    # Save checkpoint
                    ckpt = {
                        "epoch": epoch + 1,
                        # "net": net.module.state_dict(),
                        "net": net.state_dict(),
                        "adapter": adapter.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        # "scaler": scaler.state_dict(),
                        "metrics": m,
                        "args": vars(args)
                    }
                    ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth"

                    torch.save(ckpt, ckpt_path)
                    logger.info(f"💾 Saved checkpoint: {ckpt_path}")
                    ckpt["epoch"] = epoch
                    torch.save(ckpt, checkpoint_dir / f"resumed_epoch_{epoch:03d}.pt")

                    if composite > best_comp:
                        best_comp = composite
                        best_ckpt = Path(args.output_dir) / "best_model.pth"
                        torch.save(ckpt, best_ckpt)
                        logger.info(f"New best model (Composite: {best_comp:.3f})")

            dist.barrier()

        if rank == 0:
            logger.info(f"Training complete! Best composite: {best_comp:.3f}")
            logger.info(f"Checkpoints saved in {checkpoint_dir}")

    except Exception as e:
        logger.error(f"[GPU {rank}] Training failed: {e}")
        import traceback
        traceback.print_exc()
        if setup_success:
            cleanup()
        raise
    finally:
        if setup_success:
            cleanup()





def main():
    import random
    import numpy as np
    torch.manual_seed(44)
    np.random.seed(44)
    random.seed(44)
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", required=True, help="Path to EEG embeddings HDF5")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--pretrained_vd_path", required=True,
                        help="Path to vd-four-flow-v1-0-fp16-deprecated.pth")
    parser.add_argument("--image_dir", type=str,
                        default="/raid/datasets/tanaya/fm/datasets/alljoined/images",
                        help="Directory containing images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Total batch size across all GPUs (recommended: 64-128 for 4 GPUs)")
    parser.add_argument("--learning_rate", type=float, default=2e-4)# changed from 5e-5  , 1e-4
    parser.add_argument("--max_samples", type=int, default= 10000)
    parser.add_argument("--diffusion_steps", type=int, default=50,
                        help="Number of denoising steps for training")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=5,
                        help="Keep only last N epoch checkpoints (0 = keep all)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--monitor_gpu", action="store_true",
                        help="Monitor and log GPU utilization during training")
    parser.add_argument("--use_ddp", action="store_true",
                        help="Use DistributedDataParallel instead of DataParallel")

    # Adapter-specific arguments
    parser.add_argument("--eeg_dim", type=int, default=768,
                        help="Dimension of EEG embeddings")
    parser.add_argument("--spatial_size", type=int, default=16,
                        help="Spatial size for image representation (14x14 = 196 spatial dims)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    samples_per_gpu = args.batch_size // num_gpus

    logger.info("=" * 80)
    logger.info("VERSATILE DIFFUSION TRAINING WITH EEG ADAPTER")
    logger.info("=" * 80)
    logger.info(f"Training mode: {'DDP' if args.use_ddp else 'DataParallel'}")
    logger.info(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Total batch size: {args.batch_size}")
    logger.info(f"Samples per GPU: {samples_per_gpu}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"EEG embedding dimension: {args.eeg_dim}")
    logger.info(f"Spatial size: {args.spatial_size}x{args.spatial_size}")
    logger.info(f"Target image shape: (batch, 4, {args.spatial_size}, {args.spatial_size})")

    # if samples_per_gpu < 8:
    #     logger.warning("=" * 80)
    #     logger.warning(f"⚠️  WARNING: Only {samples_per_gpu} samples per GPU!")
    #     logger.warning(f"⚠️  This will result in poor GPU utilization.")
    #     logger.warning(f"⚠️  Recommended: Increase --batch_size to at least {num_gpus * 8}")
    #     logger.warning("=" * 80)

    logger.info("=" * 80)

    # # Choose training method
    # if args.use_ddp and num_gpus > 1:
    #     logger.info("Starting DDP training with EEG adapter...")
    #     mp.spawn(train_ddp, args=(num_gpus, args), nprocs=num_gpus, join=True)

    # # NEW:
    if args.use_ddp:
        logger.info("Starting DDP training with EEG adapter...")
        # Get rank and world_size from environment (set by torchrun)
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        train_ddp(rank, world_size, args)
    else:
        logger.info("Single GPU training not implemented")
    # else:
    #     if args.use_ddp and num_gpus == 1:
    #         logger.warning("DDP requested but only 1 GPU available. Using standard training.")
    #     logger.info("DataParallel mode not implemented in this version. Please use --use_ddp flag.")


if __name__ == "__main__":
    main()


