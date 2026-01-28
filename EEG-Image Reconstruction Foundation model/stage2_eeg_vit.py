
"""
EEG-ViT with Category-Aware MoE -   eeg_vit_moe_up.py with category mappings

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import h5py
import json
import logging
from tqdm import tqdm
import time
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset,DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*weights_only.*')

@dataclass
class EEGViTCNetConfig:
    """Configuration for EEGViTCNet with Self-Aware MoE"""
    # Input dimensions (flexible based on user data)
    input_trials: Optional[int] = None  # Will be set dynamically from data
    input_sequence_len: Optional[int] = None  # Will be set dynamically from data
    input_embedding_dim: int = 1024  # From harmonization layer okay done

    # TCN parameters
    tcn_channels: List[int] = None
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2

    # ViT parameters
    vit_patch_size: int = 16
    vit_embed_dim: int = 512
    vit_depth: int = 6
    vit_num_heads: int = 8
    vit_mlp_ratio: float = 4.0

    # Self-Aware MoE parameters
    # num_experts: int = 8  # Let model discover patterns
    num_experts: int = 4
    expert_capacity: int = 64
    routing_temperature: float = 1.5  #changed from 1.0
    expert_dropout: float = 0.1

    # Self-Aware Adapter parameters
    adapter_dim: int = 128
    adapter_layers: int = 2
    adaptation_strength: float = 0.1

    # Output parameters - DinoV3 compatible made
    reconstruction_dim: int = 768  # Match DinoV3-base feature dim
    dinov3_feature_dim: int = 768  # DinoV3-base (768), -large (1024), -giant (1536)

    # ADDED: Category parameters
    num_image_categories: int = 50
    num_super_categories: int = 7
    category_embedding_dim: int = 64 # changed from 128
    # category_embedding_dim: int = 128
    # ADD these new parameters:
    use_super_categories: bool = True  # Focus on super categories
    multi_dataset: bool = False  # Will be set based on input
    dataset_names: List[str] = None  # Dataset identifiers

    # Training parameters
    dropout: float = 0.1
    load_balance_weight: float = 0.1 # changed from 0.01

    def __post_init__(self):
        if self.tcn_channels is None:
            self.tcn_channels = [512, 1024, 1024, 512]


class EnhancedTCNBlock(nn.Module):
    """Enhanced TCN block """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)  # 1x1 conv
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out + residual


class TemporalConvolutionalNetwork(nn.Module):
    """new TCN """

    def __init__(self, config: EEGViTCNetConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_embedding_dim, config.tcn_channels[0])

        # TCN layers with residual connections and attention
        self.tcn_layers = nn.ModuleList()

        for i in range(len(config.tcn_channels) - 1):
            in_channels = config.tcn_channels[i]
            out_channels = config.tcn_channels[i + 1]
            dilation = 2 ** i

            tcn_block = EnhancedTCNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=config.tcn_kernel_size,
                dilation=dilation,
                dropout=config.tcn_dropout
            )
            self.tcn_layers.append(tcn_block)

        # Temporal attention for important moment detection
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.tcn_channels[-1],
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )

        # Global and local feature extraction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool = nn.AdaptiveMaxPool1d(8)  # Capture local peaks

        self.temporal_features_dim = config.tcn_channels[-1]

    def forward(self, sequence_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len, embed_dim = sequence_embeddings.shape

        # Project to TCN input dimension
        x = self.input_proj(sequence_embeddings)  # [B, seq_len, tcn_channels[0]]
        x = x.transpose(1, 2)  # [B, channels, seq_len]

        # Apply TCN layers with residual connections
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)

        # Apply temporal attention
        x_transposed = x.transpose(1, 2)  # [B, seq_len, channels]
        attn_out, _ = self.temporal_attention(x_transposed, x_transposed, x_transposed)
        x = attn_out.transpose(1, 2)  # Back to [B, channels, seq_len]

        # Extract global and local features
        global_features = self.global_pool(x).squeeze(-1)  # [B, channels]
        local_features = self.local_pool(x)  # [B, channels, 8]
        local_features = local_features.flatten(-2)  # [B, channels * 8]

        # Combine global and local information
        temporal_features = torch.cat([global_features, local_features], dim=-1)
        temporal_features = temporal_features[:, :self.temporal_features_dim]  # Truncate if needed

        # Sequence features for reconstruction
        sequence_features = x.transpose(1, 2)  # [B, seq_len, channels]

        return temporal_features, sequence_features


class TransformerBlock(nn.Module):
    """new transformer block ""

    def __init__(self, config: EEGViTCNetConfig):
        super().__init__()
        self.embed_dim = config.vit_embed_dim

        # Multi-head attention with relative position encoding
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.vit_num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        # Enhanced MLP with gating
        mlp_hidden_dim = int(self.embed_dim * config.vit_mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_hidden_dim, self.embed_dim),
            nn.Dropout(config.dropout)
        )

        # Gating mechanism for adaptive processing
        self.gate = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Gated MLP with residual
        mlp_out = self.mlp(x)
        gate_weights = self.gate(x)
        x = self.norm2(x + gate_weights * mlp_out)

        return x


class EEGViT(nn.Module):
    """ Vision Transformer """

    def __init__(self, config: EEGViTCNetConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.vit_patch_size
        self.embed_dim = config.vit_embed_dim

        # Input projection
        self.input_proj = nn.Linear(config.input_embedding_dim, self.embed_dim)

        # Positional embeddings (learnable) - will be resized dynamically
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, self.embed_dim))  # Large initial size

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.vit_depth)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def create_patches(self, sequence_embeddings: torch.Tensor) -> torch.Tensor:
        """Create patches - EXACT COPY from working version"""
        batch_size, seq_len, embed_dim = sequence_embeddings.shape

        # Pad sequence if needed
        remainder = seq_len % self.patch_size
        if remainder != 0:
            padding = self.patch_size - remainder
            sequence_embeddings = F.pad(sequence_embeddings, (0, 0, 0, padding))
            seq_len += padding

        # Create overlapping patches for better continuity
        num_patches = seq_len // self.patch_size
        patches = sequence_embeddings.view(batch_size, num_patches, self.patch_size, embed_dim)

        # Apply patch-wise attention pooling instead of simple averaging
        patch_weights = F.softmax(patches.mean(dim=-1), dim=-1).unsqueeze(-1)
        patches = (patches * patch_weights).sum(dim=2)  # Weighted average

        return patches

    def forward(self, sequence_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = sequence_embeddings.shape

        # Create patches
        patches = self.create_patches(sequence_embeddings)
        patches = self.input_proj(patches)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)

        # Dynamic positional embedding
        seq_length = x.shape[1]
        if seq_length > self.pos_embedding.shape[1]:
            # Interpolate positional embeddings if sequence is longer
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=seq_length,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_emb = self.pos_embedding[:, :seq_length, :]

        x = x + pos_emb
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        # Separate CLS and patch features
        cls_features = x[:, 0]  # [B, embed_dim]
        patch_features = x[:, 1:]  # [B, num_patches, embed_dim]

        return cls_features, patch_features


# ADDED: Category functionality
class CategoryEmbedding(nn.Module):
    """Learnable category embeddings"""

    def __init__(self, config: EEGViTCNetConfig):
        super().__init__()

        self.super_category_embedding = nn.Embedding(
            config.num_super_categories,
            config.category_embedding_dim
        )

        self.category_embedding = nn.Embedding(
            config.num_image_categories,
            config.category_embedding_dim
        )

        self.fusion = nn.Linear(config.category_embedding_dim * 2,
                                config.category_embedding_dim)

    def forward(self, super_category_ids: torch.Tensor,
                category_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        super_emb = self.super_category_embedding(super_category_ids)

        if category_ids is not None:
            category_emb = self.category_embedding(category_ids)
            combined = torch.cat([super_emb, category_emb], dim=-1)
            return self.fusion(combined)

        return super_emb


class SelfAwareExpert(nn.Module):
    """Self-aware expert """

    def __init__(self, input_dim: int, reconstruction_dim: int, expert_id: int):
        super().__init__()
        self.expert_id = expert_id
        self.input_dim = input_dim

        # Self-awareness mechanism: learn what patterns this expert handles
        self.pattern_detector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Adaptive feature transformation
        self.feature_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Specialized reconstruction decoder
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, reconstruction_dim)
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(reconstruction_dim)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-awareness: how confident is this expert for this input?
        confidence = self.pattern_detector(features)

        # Adaptive feature transformation
        adapted_features = self.feature_adapter(features)

        # Skip connection for stability
        adapted_features = features + adapted_features

        # Reconstruction
        reconstruction = self.reconstruction_decoder(adapted_features)
        reconstruction = self.output_norm(reconstruction)

        return reconstruction, confidence


class SelfAwareMixtureOfExperts(nn.Module):
    """Self-aware MoE with category awareness """

    def __init__(self, config: EEGViTCNetConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.routing_temperature = config.routing_temperature

        # Input dimension calculation -
        input_dim = config.tcn_channels[-1] + config.vit_embed_dim

        # Create self-aware experts 
        self.experts = nn.ModuleList([
            SelfAwareExpert(input_dim, config.reconstruction_dim, expert_id)
            for expert_id in range(self.num_experts)
        ])

        # ADDED: Category-aware routing
        self.category_router = nn.Sequential(
            nn.Linear(input_dim + config.category_embedding_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, self.num_experts)
        )

        # Dynamic routing network -
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(config.expert_dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Dropout(config.expert_dropout),
            nn.Linear(input_dim // 4, self.num_experts)
        )

        self._init_router_weights()

        # Temperature parameter - EXACT from working version
        #self.temperature = nn.Parameter(torch.ones(1) * 2.0)  # Start higher for exploration
        self.temperature = nn.Parameter(torch.ones(1) * 0.8)  # Start lower, not 2.0

        # CRITICAL FIX: Add noise for exploration
        self.training_noise = 0.1

        # Expert usage tracking 
        self.register_buffer('expert_usage', torch.zeros(self.num_experts))
        self.load_balance_weight = config.load_balance_weight

    def _init_router_weights(self):
        """Initialize router """
        for router in [self.router, self.category_router]:
            for layer in router:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.1)  # Small gain
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def forward(self, combined_features: torch.Tensor,
                category_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = combined_features.shape[0]

        # Add exploration noise during training
        if self.training:
            noise = torch.randn_like(combined_features) * self.training_noise
            combined_features = combined_features + noise

        # Category-aware routing if available
        if category_embeddings is not None:
            cat_input = torch.cat([combined_features, category_embeddings], dim=-1)
            category_routing = F.softmax(self.category_router(cat_input) / 0.5, dim=-1)  # Fixed temp
        else:
            category_routing = None

        # Implicit routing with lower temperature
        routing_logits = self.router(combined_features)
        current_temp = torch.clamp(self.temperature, min=0.3, max=1.0)  # Lower range
        routing_probs = F.softmax(routing_logits / current_temp, dim=-1)

        # Added epsilon to prevent zero usage
        epsilon = 0.01
        routing_probs = (1 - epsilon) * routing_probs + epsilon / self.num_experts

        # Combine category and implicit routing
        if category_routing is not None:
            alpha = 0.6
            routing_probs = alpha * category_routing + (1 - alpha) * routing_probs

        # Get expert outputs - same as before
        expert_outputs = []
        expert_confidences = []

        for expert in self.experts:
            output, confidence = expert(combined_features)
            expert_outputs.append(output)
            expert_confidences.append(confidence)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        expert_confidences = torch.stack(expert_confidences, dim=1)

        # Ensured routing weights are actually used
        routing_weights = routing_probs.unsqueeze(-1)
        confidence_weights = expert_confidences * routing_weights

        # Normalize to ensure sum = 1
        confidence_weights = F.normalize(confidence_weights, p=1, dim=1)

        final_reconstruction = torch.sum(confidence_weights * expert_outputs, dim=1)

        # Proper expert usage tracking - okay done
        with torch.no_grad():
            # Use actual routing probabilities, not averaged
            current_usage = torch.sum(routing_probs, dim=0) / batch_size
            self.expert_usage.copy_(0.95 * self.expert_usage + 0.05 * current_usage)

        #  load balancing
        load_balance_loss = self._compute_enhanced_load_balance_loss(routing_probs)

        return {
            'reconstruction': final_reconstruction,
            'routing_probs': routing_probs,
            'expert_confidences': expert_confidences.squeeze(-1),
            'expert_outputs': expert_outputs,
            'load_balance_loss': load_balance_loss,
            'expert_usage': self.expert_usage.clone()
        }

    #def _compute_load_balance_loss(self, routing_probs: torch.Tensor) -> torch.Tensor:
    def _compute_enhanced_load_balance_loss(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """ load balancing to encourage expert usage"""
        # Current usage distribution
        usage = routing_probs.mean(dim=0)

        # Penalty for unused experts
        min_usage = 1.0 / (self.num_experts * 2)  # Each expert should get at least this
        unused_penalty = F.relu(min_usage - usage).sum()

        # Standard uniform target
        target = torch.ones_like(usage) / self.num_experts
        uniform_loss = F.mse_loss(usage, target)

        total_loss = uniform_loss + 2.0 * unused_penalty
        return self.load_balance_weight * total_loss


class SelfAwareAdapter(nn.Module):
    """Self-aware adapter """

    def __init__(self, input_dim: int, config: EEGViTCNetConfig):
        super().__init__()
        self.input_dim = input_dim
        self.adapter_dim = config.adapter_dim
        self.adaptation_strength = config.adaptation_strength

        # Multi-layer adapter with residual connections
        layers = []
        current_dim = input_dim

        for _ in range(config.adapter_layers):
            layers.extend([
                nn.Linear(current_dim, self.adapter_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            current_dim = self.adapter_dim

        # Final projection back to input dimension
        layers.append(nn.Linear(self.adapter_dim, input_dim))

        self.adapter = nn.Sequential(*layers)

        # Gating mechanism for adaptive strength
        self.adaptation_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )

        # Initialize with small weights
        self._init_weights()

    def _init_weights(self):
        """Initialize adapter weights to be small"""
        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                layer.weight.data *= 0.1
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Compute adaptation
        adaptation = self.adapter(features)

        # Compute adaptive gating
        gate = self.adaptation_gate(features)

        # Apply gated adaptation with learnable strength
        adapted_features = features + self.adaptation_strength * gate * adaptation

        return adapted_features


class EEGViTCNetSelfAware(nn.Module):
    """Complete EEGViTCNet with category awareness """

    def __init__(self, config: EEGViTCNetConfig):
        super().__init__()
        self.config = config

        # Core components 
        self.tcn = TemporalConvolutionalNetwork(config)
        self.vit = EEGViT(config)

        #  Category embeddings
        self.category_embeddings = CategoryEmbedding(config)

        # Feature fusion - EXACT from working version
        combined_dim = config.tcn_channels[-1] + config.vit_embed_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(combined_dim)
        )

        # Self-aware MoE - changed to include category awareness
        self.moe = SelfAwareMixtureOfExperts(config)

        # ADDED: Category prediction heads
        self.category_predictor = nn.Sequential(
            nn.Linear(config.reconstruction_dim, config.reconstruction_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.reconstruction_dim // 2, config.num_image_categories)
        )

        self.super_category_predictor = nn.Sequential(
            nn.Linear(config.reconstruction_dim, config.reconstruction_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.reconstruction_dim // 4, config.num_super_categories)
        )

        # Subject adapters registry
        self.subject_adapters = nn.ModuleDict()
        self.combined_dim = combined_dim

    def register_subject_adapter(self, subject_id: str):
        """Register self-aware adapter """
        adapter = SelfAwareAdapter(self.combined_dim, self.config)
        self.subject_adapters[subject_id] = adapter

    def forward(self,
                sequence_embeddings: torch.Tensor,
                subject_id: Optional[str] = None,
                category_ids: Optional[torch.Tensor] = None,
                super_category_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with category support - MODIFIED from working version"""

        # Extract temporal features 
        temporal_features, sequence_features = self.tcn(sequence_embeddings)

        # Extract spatial patterns
        cls_features, patch_features = self.vit(sequence_embeddings)

        # Combine TCN and ViT features
        combined_features = torch.cat([temporal_features, cls_features], dim=-1)

        # Feature fusion 
        combined_features = self.feature_fusion(combined_features)

        # Subject adaptation 
        if subject_id and subject_id in self.subject_adapters:
            combined_features = self.subject_adapters[subject_id](combined_features)

        # Get category embeddings if available
        category_embeddings = None
        if super_category_ids is not None:
            category_embeddings = self.category_embeddings(super_category_ids, category_ids)

        # Self-aware MoE processing with category awareness
        moe_outputs = self.moe(combined_features, category_embeddings)

        #  Category predictions adding
        category_logits = self.category_predictor(moe_outputs['reconstruction'])
        super_category_logits = self.super_category_predictor(moe_outputs['reconstruction'])
        #super_category_logits = self.super_category_predictor(moe_outputs['reconstruction'])
        #super_category_probs = F.softmax(super_category_logits, dim=1)

        # Hierarchical gating: super-category confidence gates object prediction
        #gate_weights = self.category_gate(moe_outputs['reconstruction'])
        #category_logits = self.category_predictor(moe_outputs['reconstruction']) * gate_weights

        # Prepare final outputs - i modified
        outputs = {
            'reconstruction': moe_outputs['reconstruction'],
            'routing_probs': moe_outputs['routing_probs'],
            'expert_confidences': moe_outputs['expert_confidences'],
            'expert_usage': moe_outputs['expert_usage'],
            'load_balance_loss': moe_outputs['load_balance_loss'],
            'temporal_features': temporal_features,
            'spatial_features': cls_features,
            'combined_features': combined_features,
            # ADDED: Category predictions
            'predicted_categories': category_logits,
            'predicted_super_categories': super_category_logits
        }

        return outputs

    def analyze_expert_specialization(self) -> Dict[str, torch.Tensor]:
        """Analyze expert specialization - EXACT from working version"""
        analysis = {
            'expert_usage': self.moe.expert_usage,
            'routing_temperature': self.moe.temperature,
        }

        # Additional statistics
        usage = self.moe.expert_usage.cpu().numpy()
        analysis['usage_entropy'] = -np.sum(usage * np.log(usage + 1e-8))
        analysis['usage_std'] = np.std(usage)
        analysis['most_active_expert'] = int(np.argmax(usage))
        analysis['least_active_expert'] = int(np.argmin(usage))
        analysis['collapsed_experts'] = int(np.sum(usage < 0.05))  # < 5% usage

        return analysis

    def analyze_category_accuracy(correct_per_category: dict, total_per_category: dict,
                                  category_mappings: dict) -> dict:
        """Analyze per-category accuracy"""

        id_to_category = {v: k for k, v in category_mappings['super_category_to_id'].items()}

        category_accuracy = {}
        for cat_id, correct in correct_per_category.items():
            total = total_per_category.get(cat_id, 1)
            accuracy = correct / max(total, 1) * 100
            cat_name = id_to_category.get(cat_id, f"cat_{cat_id}")
            category_accuracy[cat_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }

        return category_accuracy


class CrossDatasetCategoryMapper:
    """Simple mapping preserving all existing super-categories"""

    def __init__(self):
        # NO unified mapping - keep all categories distinct
        self.category_to_id = {}

    def build_from_data(self, all_super_categories):
        """Build mapping from actual data"""
        unique_categories = list(set(all_super_categories))
        self.category_to_id = {cat: idx for idx, cat in enumerate(unique_categories)}
        return self.category_to_id

    def map_category(self, original_super_cat: str) -> dict:
        """Simple pass-through mapping"""
        return {
            'super_category': original_super_cat,
            'super_category_id': self.category_to_id.get(original_super_cat, 0)
        }

class EEGViTDataLoader:
    """ data loader with category support """

    # def __init__(self, harmonized_dir: Path, batch_size: int = 16,
    #              max_sequence_length: Optional[int] = None):
    def __init__(self, harmonized_dirs: Union[Path, List[Path]], batch_size: int = 16,
                     max_sequence_length: Optional[int] = None,
                     dataset_names: Optional[List[str]] = None,
                     use_super_categories: bool = True,
                    max_trials_per_subject: Optional[int] = None):
               if isinstance(harmonized_dirs, (str, Path)):
            self.harmonized_dirs = [Path(harmonized_dirs)]
            self.dataset_names = dataset_names or ["dataset_0"]
        else:
            self.harmonized_dirs = [Path(d) for d in harmonized_dirs]
            self.dataset_names = dataset_names or [f"dataset_{i}" for i in range(len(harmonized_dirs))]

        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.use_super_categories = use_super_categories
        self.max_trials_per_subject = max_trials_per_subject  # ADD THIS
        self.multi_dataset = len(self.harmonized_dirs) > 1

        # Find all harmonized files across datasets
        self.harmonized_files = []
        self.dataset_file_mapping = {}  # Maps file to dataset info

        # for dataset_idx, (harmonized_dir, dataset_name) in enumerate(zip(self.harmonized_dirs, self.dataset_names)):
        #     dataset_files = []
        #     # Look for .h5 files (your harmonized format)
        #     for file_path in harmonized_dir.glob("*.h5"):
        #
        #         self.harmonized_files.append(file_path)
        #         self.dataset_file_mapping[str(file_path)] = {
        #             'dataset_idx': dataset_idx,
        #             'dataset_name': dataset_name,
        #             'dataset_dir': harmonized_dir
        #         }
        #         dataset_files.append(file_path)

        for dataset_idx, (harmonized_dir, dataset_name) in enumerate(zip(self.harmonized_dirs, self.dataset_names)):
            harmonized_dir = Path(harmonized_dir)
            dataset_files = []

            # Search in subject subdirectories
            for subject_dir in harmonized_dir.iterdir():
                if subject_dir.is_dir():
                    # Look for harmonized h5 files
                    for file_path in subject_dir.glob("*_harmonized.h5"):
                        self.harmonized_files.append(file_path)
                        self.dataset_file_mapping[str(file_path)] = {
                            'dataset_idx': dataset_idx,
                            'dataset_name': dataset_name,
                            'dataset_dir': harmonized_dir
                        }
                        dataset_files.append(file_path)

            logger.info(f"Dataset '{dataset_name}': Found {len(dataset_files)} harmonized files")


        logger.info(f"Total: {len(self.harmonized_files)} harmonized files from {len(self.harmonized_dirs)} datasets")

        # Analyze data dimensions
        self._analyze_data_dimensions()

        # Build unified category mappings across all datasets
        self.category_mappings = {'category_to_id': {}, 'super_category_to_id': {}}
        self._build_unified_category_mappings()



     def _build_unified_category_mappings(self):
        """Build unified category mappings across all datasets focusing on super_categories"""
        logger.info("Building unified category mappings across datasets...")

        all_super_categories = set()
        all_image_categories = set()
        dataset_category_stats = {}

        for dataset_name in self.dataset_names:
            dataset_files = [f for f in self.harmonized_files
                             if self.dataset_file_mapping[str(f)]['dataset_name'] == dataset_name]

            dataset_super_cats = set()
            dataset_image_cats = set()

            # Sample files from this dataset to extract categories
            sample_files = dataset_files[:3] if len(dataset_files) > 3 else dataset_files

            for file_path in sample_files:
                try:
                    with h5py.File(file_path, 'r') as f:
                        # Get super categories
                        if 'super_categories' in f:
                            super_cats = [cat.decode('utf-8') if isinstance(cat, bytes) else str(cat)
                                          for cat in f['super_categories'][:]]
                            dataset_super_cats.update(super_cats)
                            all_super_categories.update(super_cats)

                        # Get image categories
                        if 'image_categories' in f:
                            img_cats = [cat.decode('utf-8') if isinstance(cat, bytes) else str(cat)
                                        for cat in f['image_categories'][:]]
                            dataset_image_cats.update(img_cats)
                            all_image_categories.update(img_cats)

                except Exception as e:
                    logger.warning(f"Could not read categories from {file_path}: {e}")
                    continue

            dataset_category_stats[dataset_name] = {
                'super_categories': sorted(list(dataset_super_cats)),
                'image_categories': sorted(list(dataset_image_cats)),
                'n_files': len(dataset_files)
            }

        # Create unified mappings
        unified_super_categories = sorted(list(all_super_categories))
        unified_image_categories = sorted(list(all_image_categories))

        # Build ID mappings
        for idx, super_cat in enumerate(unified_super_categories):
            self.category_mappings['super_category_to_id'][super_cat] = idx

        for idx, img_cat in enumerate(unified_image_categories):
            self.category_mappings['category_to_id'][img_cat] = idx

        # Print summary
        logger.info(f"\n=== UNIFIED CATEGORY ANALYSIS ===")
        logger.info(f"Super Categories ({len(unified_super_categories)}): {unified_super_categories}")
        logger.info(f"Image Categories: {len(unified_image_categories)} total")

        for dataset_name, stats in dataset_category_stats.items():
            logger.info(f"\n{dataset_name} Dataset:")
            logger.info(f"  Super Categories: {len(stats['super_categories'])} - {stats['super_categories']}")
            logger.info(f"  Image Categories: {len(stats['image_categories'])}")
            logger.info(f"  Files: {stats['n_files']}")

        if self.use_super_categories:
            logger.info(f"\n FOCUSING ON SUPER CATEGORIES: {len(unified_super_categories)} categories")
        else:
            logger.info(f"\ FOCUSING ON IMAGE CATEGORIES: {len(unified_image_categories)} categories")


    def _analyze_data_dimensions(self):
        """Analyze data dimensions"""
        if self.harmonized_files:
            max_seq_len = 0
            for file_path in self.harmonized_files:
                with h5py.File(file_path, 'r') as f:
                    sample_data = f['embeddings_sequence']
                    n_trials, seq_len, embed_dim = sample_data.shape
                    max_seq_len = max(max_seq_len, seq_len)

            with h5py.File(self.harmonized_files[0], 'r') as f:
                sample_data = f['embeddings_sequence']
                total_trials, _, self.embedding_dim = sample_data.shape

                # APPLY LIMIT HERE
                if self.max_trials_per_subject:
                    self.n_trials = min(total_trials, self.max_trials_per_subject)
                else:
                    self.n_trials = total_trials

                self.sequence_length = max_seq_len

            logger.info(f"Data dimensions: {self.n_trials} trials (limited from {total_trials}), "
                        f"{self.sequence_length} seq_len, {self.embedding_dim} embed_dim")

    def load_batch(self, file_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Load batch - EXACT from working version"""
        batch_data = []
        batch_subjects = []

        for idx in file_indices:
            file_path = self.harmonized_files[idx]

            with h5py.File(file_path, 'r') as f:
                # Load the harmonized embeddings
                data = f['embeddings_sequence'][:]  # Shape: (n_trials, seq_len, 1024)
                subject_id = f.attrs.get('subject_id', f'sub-{idx}')

                # Truncate sequence if needed
                if self.max_sequence_length and data.shape[1] > self.max_sequence_length:
                    data = data[:, :self.max_sequence_length, :]

                batch_data.append(torch.from_numpy(data).float())
                batch_subjects.append(subject_id)

        return {
            'data': torch.stack(batch_data, dim=0),  # [batch_size, n_trials, seq_len, 1024]
            'subject_ids': batch_subjects
        }

     def get_sample_with_categories(self, file_path: Path, trial_idx: int):
        """Use super_category string directly instead of  IDs"""
        try:
            with h5py.File(file_path, 'r') as f:
                if trial_idx >= f['embeddings_sequence'].shape[0]:
                    return None

                sequence_emb = f['embeddings_sequence'][trial_idx]
                subject_id = f.attrs.get('subject_id', 'unknown')

                sample = {
                    'sequence_embeddings': torch.from_numpy(sequence_emb).float(),
                    'subject_id': subject_id,
                    'trial_idx': trial_idx
                }

                # CRITICAL FIX: Use string categories directly
                if 'super_categories' in f and trial_idx < len(f['super_categories']):
                    super_cat_raw = f['super_categories'][trial_idx]
                    super_cat = super_cat_raw.decode('utf-8') if isinstance(super_cat_raw, bytes) else str(
                        super_cat_raw)

                    # Convert to ID using your existing mapping
                    super_cat_id = self.category_mappings['super_category_to_id'].get(super_cat, -1)

                    sample.update({
                        'super_category': super_cat,
                        'super_category_id': super_cat_id,
                        'mapped': super_cat_id >= 0
                    })
                else:
                    sample.update({
                        'super_category': 'unknown',
                        'super_category_id': -1,
                        'mapped': False
                    })

                return sample

        except Exception as e:
            logger.error(f"Error loading sample: {e}")
            return None

    def get_data_config(self) -> Dict[str, int]:
        """Get data config with unified category info"""

        # Get dimensions from first file if not already set
        if not hasattr(self, 'embedding_dim'):
            with h5py.File(self.harmonized_files[0], 'r') as f:
                sample_data = f['embeddings_sequence']
                _, self.sequence_length, self.embedding_dim = sample_data.shape

        config = {
            'input_sequence_len': self.sequence_length,  # Added this back
            'input_embedding_dim': self.embedding_dim,
            'num_image_categories': len(self.category_mappings['category_to_id']),
            'num_super_categories': len(self.category_mappings['super_category_to_id']),
            'use_super_categories': self.use_super_categories,
            'multi_dataset': self.multi_dataset,
            'dataset_names': self.dataset_names
        }

        return config


class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss for handling class imbalance"""

    def __init__(self, num_classes, samples_per_class=None, beta=0.9999, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma

        # Compute effective number of samples
        if samples_per_class is not None:
            effective_num = 1.0 - np.power(beta, samples_per_class)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / weights.sum() * num_classes
            self.class_weights = torch.FloatTensor(weights)
        else:
            self.class_weights = torch.ones(num_classes)

    def forward(self, logits, targets):
        """
        logits: [B, num_classes]
        targets: [B]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss

        # FIXED: Move class_weights to correct device before indexing
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)

        weights = self.class_weights[targets]
        cb_focal_loss = (weights * focal_loss).mean()

        return cb_focal_loss

from torch.cuda.amp import autocast, GradScaler



class MultiGPUEEGViTTrainer:
    """Multi-GPU trainer """

    def __init__(self, model: EEGViTCNetSelfAware, config: EEGViTCNetConfig):
        self.config = config

        self.device = torch.device('cuda:0')
        self.model = model.to(self.device)  # Single GPU only

        logger.info(f"Using single GPU: cuda:0")
        logger.info("Increase batch_size to 128 to fully utilize single GPU")


        #Setup device and DataParallel before moving model
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

         self.optimizer = torch.optim.AdamW(
            self.model.parameters(),  # This was correct
            lr=1e-5,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )

        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.feature_loss = nn.CosineSimilarity(dim=-1)

        # CB-Focal Loss - move to device
        self.cb_focal_loss_super = CBFocalLoss(
            num_classes=config.num_super_categories,
            beta=0.9999,
            gamma=2.0
        ).to(self.device)

        self.cb_focal_loss_img = CBFocalLoss(
            num_classes=config.num_image_categories,
            beta=0.9999,
            gamma=2.0
        ).to(self.device)

        # Mixed precision
        self.scaler = GradScaler('cuda')
        self.use_amp = False

        # MoE annealing
        self.moe_temp_schedule = {
            'initial_temp': 1.5,
            'final_temp': 0.3,
            'anneal_epochs': 50,
            'strategy': 'cosine'
        }
    def anneal_moe_temperature(self, epoch):
        """Anneal MoE routing temperature over training"""
        if epoch >= self.moe_temp_schedule['anneal_epochs']:
            return  # Already annealed

        initial = self.moe_temp_schedule['initial_temp']
        final = self.moe_temp_schedule['final_temp']
        progress = epoch / self.moe_temp_schedule['anneal_epochs']

        if self.moe_temp_schedule['strategy'] == 'cosine':
            # Cosine annealing
            new_temp = final + 0.5 * (initial - final) * (1 + np.cos(np.pi * progress))
        else:
            # Linear annealing
            new_temp = initial + (final - initial) * progress

        # Update temperature
        moe = self.model.module.moe if hasattr(self.model, 'module') else self.model.moe
        with torch.no_grad():
            moe.temperature.data.fill_(new_temp)

        logger.info(f"MoE temperature annealed to: {new_temp:.3f}")

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                     targets: Optional[torch.Tensor] = None,
                     category_ids: Optional[torch.Tensor] = None,
                     super_category_ids: Optional[torch.Tensor] = None,
                     mapped_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute loss with category support - CLEANED UP"""
        # print(f"DEBUG - Outputs keys: {outputs.keys()}")
        # print(f"DEBUG - Super category IDs: {super_category_ids}")
        # print(f"DEBUG - Reconstruction shape: {outputs['reconstruction'].shape}")
        # Reconstruction loss
        if targets is None:
            reconstruction_target = outputs['temporal_features'].detach()
            reconstruction_loss = self.reconstruction_loss(
                outputs['reconstruction'][:, :reconstruction_target.shape[1]],
                reconstruction_target
            )
        else:
            reconstruction_loss = self.reconstruction_loss(outputs['reconstruction'], targets)

        ####new added
        # Category losses using CB-Focal Loss
        category_loss = torch.tensor(0.0, device=self.device)
        super_category_loss = torch.tensor(0.0, device=self.device)
        cb_focal_super_loss = torch.tensor(0.0, device=self.device)

        # Super category loss with CB-Focal
        if super_category_ids is not None:
            valid_mask = super_category_ids >= 0
            if valid_mask.sum() > 0:
                valid_predictions = outputs['predicted_super_categories'][valid_mask]
                valid_targets = super_category_ids[valid_mask]

                # Standard CE loss
                super_category_loss = F.cross_entropy(valid_predictions, valid_targets)

                # CB-Focal loss
                cb_focal_super_loss = self.cb_focal_loss_super(valid_predictions, valid_targets)

        # Image category loss with CB-Focal
        if category_ids is not None:
            valid_mask = category_ids >= 0
            if valid_mask.sum() > 0:
                valid_predictions = outputs['predicted_categories'][valid_mask]
                valid_targets = category_ids[valid_mask]
                category_loss = self.cb_focal_loss_img(valid_predictions, valid_targets)



        # Load balance and specialization losses
        load_balance_loss = outputs.get('load_balance_loss', 0)
        if torch.is_tensor(load_balance_loss):
            load_balance_loss = load_balance_loss.mean()

        routing_probs = outputs['routing_probs']
        specialization_loss = -torch.mean(torch.sum(routing_probs * torch.log(routing_probs + 1e-8), dim=-1))

        # FIXED: Balanced loss weights
        # total_loss = (
        #         reconstruction_loss +
        #         0.01 * load_balance_loss +  # Reasonable load balance
        #         0.001 * specialization_loss +  # Small specialization
        #         0.01 * category_loss +  # Low image category weight
        #         0.1 * super_category_loss  # Higher but reasonable super category weight
        # )

        # Updated total loss with CB-Focal
        total_loss = (
                reconstruction_loss +
                0.01 * load_balance_loss +
                0.001 * specialization_loss +
                0.01 * category_loss +  # Image category (CB-Focal)
                0.1 * super_category_loss +  # Super category (CE)
                0.5 * cb_focal_super_loss  # Super category (CB-Focal) - main focus
        )

        return {
            'total_loss': total_loss.mean() if torch.is_tensor(total_loss) and total_loss.dim() > 0 else total_loss,
            'reconstruction_loss': reconstruction_loss.mean() if torch.is_tensor(
                reconstruction_loss) and reconstruction_loss.dim() > 0 else reconstruction_loss,
            'load_balance_loss': load_balance_loss,
            'specialization_loss': specialization_loss.mean() if torch.is_tensor(
                specialization_loss) and specialization_loss.dim() > 0 else specialization_loss,
            'category_loss': category_loss.mean() if torch.is_tensor(
                category_loss) and category_loss.dim() > 0 else category_loss,
            'super_category_loss': super_category_loss.mean() if torch.is_tensor(
                super_category_loss) and super_category_loss.dim() > 0 else super_category_loss,
            'cb_focal_super_loss': cb_focal_super_loss.mean() if torch.is_tensor(
                cb_focal_super_loss) and cb_focal_super_loss.dim() > 0 else cb_focal_super_loss
            }

    def train_step(self, batch_data: torch.Tensor,
                   targets: Optional[torch.Tensor] = None,
                   subject_ids: Optional[List[str]] = None,
                   category_ids: Optional[torch.Tensor] = None,
                   super_category_ids: Optional[torch.Tensor] = None,
                   mapped_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Single training step - MODIFIED from working version"""

        self.model.train()
        self.optimizer.zero_grad()

        # Move data to GPU
        batch_data = batch_data.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)
        if category_ids is not None:
            category_ids = category_ids.to(self.device)
        if super_category_ids is not None:
            super_category_ids = super_category_ids.to(self.device)
        if mapped_mask is not None:
            mapped_mask = mapped_mask.to(self.device)

        # Forward pass with category info
        # outputs = self.model(batch_data, subject_ids[0] if subject_ids else None,
        #                      category_ids, super_category_ids)
        #
        # # Compute loss with category info
        # loss_dict = self.compute_loss(outputs, targets, category_ids, super_category_ids, mapped_mask)
        #
        # # Backward pass 
        # loss_dict['total_loss'].backward()
        # new
        with autocast(enabled=self.use_amp):
            outputs = self.model(batch_data, subject_ids[0] if subject_ids else None,category_ids, super_category_ids)
            loss_dict = self.compute_loss(outputs, targets, category_ids, super_category_ids, mapped_mask)

        self.scaler.scale(loss_dict['total_loss']).backward()
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping for stability
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)  # changed  Much stricter

        # self.optimizer.step()
        # self.scheduler.step()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.scheduler.step()

        return {k: v.item() if torch.is_tensor(v) else v
                for k, v in loss_dict.items()}


def evaluate_super_category_accuracy(trainer, model, data_loader, device, num_samples=1000):
    """Evaluate super-category accuracy using harmonized data"""
    model.eval()

    super_cat_correct = 0
    total_valid = 0
    super_cat_distribution = {}

    with torch.no_grad():
        for file_path in data_loader.harmonized_files[:5]:
            with h5py.File(file_path, 'r') as f:
                data = f['embeddings_sequence'][:]

            for trial_idx in range(min(50, data.shape[0])):  # More trials per file
                sample = data_loader.get_sample_with_categories(file_path, trial_idx)
                if not sample or not sample['mapped']:
                    continue

                trial_data = sample['sequence_embeddings'].unsqueeze(0).to(device)
                if trial_data.shape[1] < model.config.input_sequence_len:
                    padding = model.config.input_sequence_len - trial_data.shape[1]
                    trial_data = F.pad(trial_data, (0, 0, 0, padding))

                outputs = model(trial_data)

                # Super-category prediction
                super_pred = torch.argmax(outputs['predicted_super_categories'], dim=1)
                true_super_id = sample['super_category_id']

                if super_pred.item() == true_super_id:
                    super_cat_correct += 1

                # Track distribution
                super_cat = sample['super_category']
                super_cat_distribution[super_cat] = super_cat_distribution.get(super_cat, 0) + 1

                total_valid += 1

                if total_valid >= num_samples:
                    break

    super_accuracy = super_cat_correct / max(total_valid, 1) * 100

    logger.info(f"Super-Category Accuracy: {super_accuracy:.1f}% ({super_cat_correct}/{total_valid})")
    logger.info(f"Super-category distribution: {super_cat_distribution}")

    return {
        'super_category_accuracy': super_accuracy,
        'total_samples': total_valid,
        'distribution': super_cat_distribution
    }

def create_eegvitcnet_selfaware(
        input_sequence_len: Optional[int] = None,
        input_embedding_dim: int = 1024,
        num_experts: int = 8,
        dinov3_size: str = 'base',
        num_image_categories: int = 50,
        num_super_categories: int = 7
) -> EEGViTCNetSelfAware:
    """Factory function """

    # Use input embedding dimension for reconstruction 
    reconstruction_feature_dim = input_embedding_dim  # 1024, matches my data

    config = EEGViTCNetConfig(
        input_sequence_len=input_sequence_len,
        input_embedding_dim=input_embedding_dim,
        num_experts=num_experts,
        reconstruction_dim=reconstruction_feature_dim,  # This sets it to 1024
        dinov3_feature_dim=reconstruction_feature_dim,
        # Category parameters
        num_image_categories=num_image_categories,
        num_super_categories=num_super_categories
    )

    return EEGViTCNetSelfAware(config)


def main_training_pipeline(data_loader,
                           output_dir: str,
                           num_epochs: int = 100,
                           batch_size: int = 8,
                           num_experts: int = 4,
                           dinov3_size: str = 'base'):
    """Main training pipeline """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting EEGViTCNet Category-Aware MoE Training Pipeline")
    logger.info(f"Output path: {output_path}")

    # Get data configuration
    data_config = data_loader.get_data_config()
    logger.info(f"Data configuration: {data_config}")

    # Create model with dynamic configuration - MODIFIED to include categories
    model = create_eegvitcnet_selfaware(
        input_sequence_len=data_config['input_sequence_len'],
        input_embedding_dim=data_config['input_embedding_dim'],
        num_experts=num_experts,
        dinov3_size=dinov3_size,
        num_image_categories=data_config['num_image_categories'],
        num_super_categories=data_config['num_super_categories']
    )

    # Create trainer
    trainer = MultiGPUEEGViTTrainer(model, model.config)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    return model, trainer


# Copy all utility functions from working version exactly
def create_subject_splits(harmonized_files, split_ratio=0.9):
    """Split files by subjects - EXACT from working version"""
    subjects = {}
    for file_path in harmonized_files:
        with h5py.File(file_path, 'r') as f:
            subject_id = f.attrs.get('subject_id', 'unknown')
            if subject_id not in subjects:
                subjects[subject_id] = []
            subjects[subject_id].append(file_path)

    subject_list = list(subjects.keys())
    train_size = int(len(subject_list) * split_ratio)

    train_subjects = subject_list[:train_size]
    val_subjects = subject_list[train_size:]

    train_files = [f for s in train_subjects for f in subjects[s]]
    val_files = [f for s in val_subjects for f in subjects[s]]

    logger.info(f"Split: {len(train_subjects)} train subjects, {len(val_subjects)} val subjects")
    return train_files, val_files


def parse_subject_list(subjects_str):
    """Parse subject list - EXACT from working version"""
    if not subjects_str:
        return []

    subjects = []
    for part in subjects_str.split(','):
        part = part.strip()
        if '-' in part and not part.startswith('sub-'):
            start, end = map(int, part.split('-'))
            subjects.extend([f'sub-{i:02d}' for i in range(start, end + 1)])
            subjects.extend([f'sub-{i:03d}' for i in range(start, end + 1)])
        else:
            if part.isdigit():
                subjects.append(f'sub-{int(part):02d}')
                subjects.append(f'sub-{int(part):03d}')
            else:
                subjects.append(part)

    return subjects


def filter_subject_files_by_dataset(harmonized_files, train_dataset=None, val_dataset=None,
                                    train_subjects_str=None, val_subjects_str=None):
    """Enhanced filtering - EXACT from working version"""
    datasets = {}
    for file_path in harmonized_files:
        dataset_name = file_path.parent.name
        with h5py.File(file_path, 'r') as f:
            subject_id = f.attrs.get('subject_id', 'unknown')
        if dataset_name not in datasets:
            datasets[dataset_name] = {}
        if subject_id not in datasets[dataset_name]:
            datasets[dataset_name][subject_id] = []
        datasets[dataset_name][subject_id].append(file_path)

    train_files = []
    val_files = []

    if train_dataset and val_dataset:
        if train_dataset in datasets:
            train_files = [f for subjects in datasets[train_dataset].values() for f in subjects]
        if val_dataset in datasets:
            val_files = [f for subjects in datasets[val_dataset].values() for f in subjects]
    elif train_subjects_str or val_subjects_str:
        train_subject_ids = parse_subject_list(train_subjects_str) if train_subjects_str else []
        val_subject_ids = parse_subject_list(val_subjects_str) if val_subjects_str else []

        for dataset_name, subjects in datasets.items():
            for subject_id in subjects:
                if subject_id in train_subject_ids:
                    train_files.extend(subjects[subject_id])
                elif subject_id in val_subject_ids:
                    val_files.extend(subjects[subject_id])

    return train_files, val_files



    # Add this temporary debug function:
    # def debug_h5_file(file_path, trial_idx=0):
    #     with h5py.File(file_path, 'r') as f:
    #         print(f"DEBUG H5 keys: {list(f.keys())}")
    #         if 'super_categories' in f:
    #             print(f"super_categories[{trial_idx}]: {f['super_categories'][trial_idx]}")
    #         if 'super_category_ids' in f:
    #             print(f"super_category_ids[{trial_idx}]: {f['super_category_ids'][trial_idx]}")
    #         if 'image_categories' in f:
    #             print(f"image_categories[{trial_idx}]: {f['image_categories'][trial_idx]}")
    #
    # # Call this in your training loop:
    # debug_h5_file(batch_files[0], 0)

def evaluate_hierarchical_accuracy(self, model, data_loader, device, num_samples=1000):
    """Evaluate both super-category and object-level accuracy"""
    model.eval()

    super_cat_correct = 0
    object_correct = 0
    total_mapped = 0

    with torch.no_grad():
        for file_path in data_loader.harmonized_files[:5]:
            with h5py.File(file_path, 'r') as f:
                data = f['embeddings_sequence'][:]

            for trial_idx in range(min(20, data.shape[0])):
                sample = data_loader.get_sample_with_categories(file_path, trial_idx)
                if not sample or not sample['mapped']:
                    continue

                trial_data = sample['sequence_embeddings'].unsqueeze(0).to(device)
                outputs = model(trial_data)

                # Super-category accuracy
                super_pred = torch.argmax(outputs['predicted_super_categories'], dim=1)
                if super_pred.item() == sample['super_category_id']:
                    super_cat_correct += 1

                # Object-level accuracy (only if super-category is correct)
                if super_pred.item() == sample['super_category_id']:
                    obj_pred = torch.argmax(outputs['predicted_categories'], dim=1)
                    if obj_pred.item() == sample['category_id']:
                        object_correct += 1

                total_mapped += 1

                if total_mapped >= num_samples:
                    break

    super_accuracy = super_cat_correct / max(total_mapped, 1) * 100
    object_accuracy = object_correct / max(super_cat_correct, 1) * 100  # Conditional accuracy

    logger.info(f"Hierarchical Accuracy:")
    logger.info(f"  Super-category: {super_accuracy:.1f}% ({super_cat_correct}/{total_mapped})")
    logger.info(f"  Object (given correct super-cat): {object_accuracy:.1f}% ({object_correct}/{super_cat_correct})")

    return {
        'super_category_accuracy': super_accuracy,
        'conditional_object_accuracy': object_accuracy,
        'total_samples': total_mapped
    }


def validate_epoch(model, trainer, data_loader, val_files, batch_size=4):
    """Validate epoch ""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(val_files), batch_size):
            batch_files = val_files[i:i + batch_size]

            batch_data = []
            batch_subjects = []

            for file_path in batch_files:
                with h5py.File(file_path, 'r') as f:
                    data = f['embeddings_sequence'][:]  # Shape: (n_trials, seq_len, embed_dim)
                    subject_id = f.attrs.get('subject_id', f'sub-{i}')

                    # Check actual data dimensions
                    actual_trials = data.shape[0]  # This might be different per file

                    if data.shape[1] < model.config.input_sequence_len:
                        padding = model.config.input_sequence_len - data.shape[1]
                        data = np.pad(data, ((0, 0), (0, padding), (0, 0)), mode='constant')

                    batch_data.append(torch.from_numpy(data).float())
                    batch_subjects.append(subject_id)

            if not batch_data:
                continue

            #  Use the minimum number of trials across all files in batch
            min_trials = min([data.shape[0] for data in batch_data])

            # Process only up to the minimum number of trials
            for trial_idx in range(min_trials):
                try:
                    trial_batch = torch.stack([data[trial_idx] for data in batch_data])
                    trial_batch = trial_batch.to(trainer.device)

                    outputs = trainer.model(trial_batch, batch_subjects[0])
                    loss_dict = trainer.compute_loss(outputs)
                    total_loss += loss_dict['total_loss'].item()
                    num_batches += 1

                except Exception as e:
                    logger.warning(f"Error in validation trial {trial_idx}: {e}")
                    continue

    model.train()
    return total_loss / max(num_batches, 1)

def save_enhanced_checkpoint(model, trainer, data_loader, checkpoint_path):
    """Save checkpoint with category mappings"""
    if isinstance(trainer.model, nn.DataParallel):
        model_state = trainer.model.module.state_dict()
    else:
        model_state = trainer.model.state_dict()

    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'config': model.config.__dict__,
        # Catefory: Save category mappings
        'category_mappings': data_loader.category_mappings,
        'category_vocab': {
            'image_categories': list(data_loader.category_mappings['category_to_id'].keys()),
            'super_categories': list(data_loader.category_mappings['super_category_to_id'].keys())
        },
        'training_metadata': {
            'expert_usage': model.moe.expert_usage.cpu().numpy().tolist(),
            'routing_temperature': model.moe.temperature.item(),
            'num_image_categories': len(data_loader.category_mappings['category_to_id']),
            'num_super_categories': len(data_loader.category_mappings['super_category_to_id']),

            # ADD: Data source information
            'data_source': 'pt_shards',  # vs 'h5_files'
            'training_data_format': 'sharded_pt'
        },
        # ADD: Dataset names for multi-dataset scenarios
        'dataset_names': data_loader.dataset_names if hasattr(data_loader, 'dataset_names') else ['unknown']

    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f" Checkpoint saved to: {checkpoint_path}")
    logger.info(f"Saved {len(checkpoint['category_vocab']['image_categories'])} image categories")
    logger.info(f"Saved {len(checkpoint['category_vocab']['super_categories'])} super categories")


def visualize_enhanced_tsne_with_labels(model, data_loader, device, val_files=None,
                                        n_samples=1000, save_path=None):
    """Enhanced t-SNE with category labels and better visualization"""
    model.eval()
    embeddings = []
    subject_labels = []
    category_labels = []
    super_category_labels = []
    expert_assignments = []  # Track which expert was most active

    with torch.no_grad():
        files_to_process = val_files if val_files else data_loader.harmonized_files[:3]  # Limit files for speed

        for file_path in files_to_process:
            if len(embeddings) >= n_samples:
                break

            with h5py.File(file_path, 'r') as f:
                subject_id = f.attrs.get('subject_id', 'unknown')
                data = f['embeddings_sequence'][:]

            for trial_idx in range(min(20, data.shape[0])):  # Limit trials per file
                if len(embeddings) >= n_samples:
                    break

                sample = data_loader.get_sample_with_categories(file_path, trial_idx)
                if sample is None:
                    continue

                trial_data = sample['sequence_embeddings'].unsqueeze(0).to(device)

                if trial_data.shape[1] < model.config.input_sequence_len:
                    padding = model.config.input_sequence_len - trial_data.shape[1]
                    trial_data = F.pad(trial_data, (0, 0, 0, padding))

                outputs = model(trial_data)
                z = outputs['combined_features']
                routing_probs = outputs['routing_probs']

                # Find most active expert
                most_active_expert = torch.argmax(routing_probs, dim=1).item()

                embeddings.append(z.cpu().numpy())
                subject_labels.append(subject_id)
                category_labels.append(sample.get('image_category', 'unknown'))
                super_category_labels.append(sample.get('super_category', 'unknown'))
                expert_assignments.append(most_active_expert)

    if len(embeddings) < 10:
        print(f"Only collected {len(embeddings)} samples")
        return

    # Convert to arrays
    embeddings = np.vstack(embeddings)
    category_labels = np.array(category_labels)
    super_category_labels = np.array(super_category_labels)
    expert_assignments = np.array(expert_assignments)

    # t-SNE
    print(f"Running t-SNE on {len(embeddings)} embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4))
    reduced = tsne.fit_transform(embeddings)

    # Create enhanced 2x2 plot with labels
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 20))

    # Plot 1: Top Super Categories with Labels
    unique_super_cats = np.unique(super_category_labels)[:10]  # Top 10 most common
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_super_cats)))

    for i, super_cat in enumerate(unique_super_cats):
        if super_cat == 'unknown':
            continue
        mask = super_category_labels == super_cat
        if np.any(mask):
            ax1.scatter(reduced[mask, 0], reduced[mask, 1],
                        c=[colors[i]], label=super_cat, alpha=0.7, s=40)

    ax1.set_title("EEG Embeddings by Super Category (Top 10)", fontsize=16, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Expert Assignments (MoE Visualization)
    expert_colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for expert_id in range(4):
        mask = expert_assignments == expert_id
        if np.any(mask):
            ax2.scatter(reduced[mask, 0], reduced[mask, 1],
                        c=[expert_colors[expert_id]],
                        label=f'Expert {expert_id}', alpha=0.7, s=40)

    ax2.set_title("EEG Embeddings by MoE Expert Assignment", fontsize=16, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Top Image Categories with Labels
    # Get most frequent categories
    unique_categories, counts = np.unique(category_labels, return_counts=True)
    top_categories = unique_categories[np.argsort(counts)[-15:]]  # Top 15

    colors = plt.cm.viridis(np.linspace(0, 1, len(top_categories)))

    for i, category in enumerate(top_categories):
        if category == 'unknown':
            continue
        mask = category_labels == category
        if np.any(mask):
            ax3.scatter(reduced[mask, 0], reduced[mask, 1],
                        c=[colors[i]], label=category[:12], alpha=0.7, s=40)  # Truncate long names

    ax3.set_title("EEG Embeddings by Image Category (Top 15)", fontsize=16, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Expert Usage Analysis
    expert_usage_counts = np.bincount(expert_assignments, minlength=8)
    bars = ax4.bar(range(8), expert_usage_counts, color=expert_colors)
    ax4.set_title("Expert Usage Distribution", fontsize=16, fontweight='bold')
    ax4.set_xlabel("Expert ID")
    ax4.set_ylabel("Number of Samples")
    ax4.grid(True, alpha=0.3)

    # Add percentage labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        pct = (height / len(embeddings)) * 100
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Print enhanced statistics
    print(f"\n=== ENHANCED ANALYSIS SUMMARY ===")
    print(f"Total samples analyzed: {len(embeddings)}")
    print(f"Unique super categories: {len(unique_super_cats)}")
    print(f"Unique image categories: {len(unique_categories)}")
    print(f"Expert usage distribution: {expert_usage_counts}")
    print(f"Most active expert: Expert {np.argmax(expert_usage_counts)}")
    print(f"Category diversity: {len(unique_categories) / len(embeddings) * 100:.1f}%")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved enhanced plot to {save_path}")

    plt.show()

    return {
        'expert_usage': expert_usage_counts,
        'top_categories': top_categories,
        'super_categories': unique_super_cats
    }


def visualize_moe_routing_patterns(model, data_loader, device, n_samples=500):
    """Visualize how MoE routing changes with different inputs"""
    model.eval()

    routing_patterns = []
    category_info = []

    with torch.no_grad():
        sample_count = 0
        for file_path in data_loader.harmonized_files[:2]:
            if sample_count >= n_samples:
                break

            with h5py.File(file_path, 'r') as f:
                data = f['embeddings_sequence'][:]

            for trial_idx in range(min(10, data.shape[0])):
                if sample_count >= n_samples:
                    break

                sample = data_loader.get_sample_with_categories(file_path, trial_idx)
                if sample is None:
                    continue

                trial_data = sample['sequence_embeddings'].unsqueeze(0).to(device)

                if trial_data.shape[1] < model.config.input_sequence_len:
                    padding = model.config.input_sequence_len - trial_data.shape[1]
                    trial_data = F.pad(trial_data, (0, 0, 0, padding))

                outputs = model(trial_data)
                routing_probs = outputs['routing_probs'].cpu().numpy()[0]

                routing_patterns.append(routing_probs)
                category_info.append(sample.get('super_category', 'unknown'))
                sample_count += 1

    routing_patterns = np.array(routing_patterns)
    category_info = np.array(category_info)

    # Create routing pattern visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Average routing by category
    unique_cats = np.unique(category_info)[:4]  # Top 8 categories
    avg_routing_by_cat = {}

    for cat in unique_cats:
        if cat == 'unknown':
            continue
        mask = category_info == cat
        if np.any(mask):
            avg_routing_by_cat[cat] = np.mean(routing_patterns[mask], axis=0)

    if avg_routing_by_cat:
        categories = list(avg_routing_by_cat.keys())
        routing_matrix = np.array(list(avg_routing_by_cat.values()))

        im1 = ax1.imshow(routing_matrix, cmap='viridis', aspect='auto')
        ax1.set_title("Average Expert Routing by Category", fontweight='bold')
        ax1.set_xlabel("Expert ID")
        ax1.set_ylabel("Category")
        ax1.set_yticks(range(len(categories)))
        ax1.set_yticklabels([cat[:15] for cat in categories])
        ax1.set_xticks(range(4))
        plt.colorbar(im1, ax=ax1)

    # Plot 2: Routing probability distribution
    ax2.boxplot([routing_patterns[:, i] for i in range(4)],
                labels=[f'Expert {i}' for i in range(4)])
    ax2.set_title("Expert Routing Probability Distribution", fontweight='bold')
    ax2.set_ylabel("Routing Probability")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Specialization measure
    # Calculate how "specialized" each expert is (higher entropy = less specialized)
    specialization_scores = []
    for i in range(4):
        expert_probs = routing_patterns[:, i]
        # Higher values = more specialized (concentrated on fewer samples)
        specialization = np.var(expert_probs)
        specialization_scores.append(specialization)

    bars = ax3.bar(range(4), specialization_scores, color=plt.cm.tab10(np.linspace(0, 1, 4)))
    ax3.set_title("Expert Specialization Scores", fontweight='bold')
    ax3.set_xlabel("Expert ID")
    ax3.set_ylabel("Specialization (Variance)")
    ax3.grid(True, alpha=0.3)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    print("\n=== MoE ROUTING ANALYSIS ===")
    print(
        f"Routing entropy (lower = more specialized): {-np.mean(np.sum(routing_patterns * np.log(routing_patterns + 1e-8), axis=1)):.3f}")
    print(f"Most specialized expert: Expert {np.argmax(specialization_scores)}")
    print(f"Average routing probabilities: {np.mean(routing_patterns, axis=0)}")

    return routing_patterns, category_info




class EEGH5Dataset(Dataset):
    def __init__(self, file_paths, data_loader, max_trials_per_file=5000, seq_len=None):
        self.samples = []
        self.file_handles = {}
        self.seq_len = seq_len
        self.data_loader = data_loader

        for file_path in file_paths:
            # Keep files open with memory mapping
            h5_file = h5py.File(file_path, "r", rdcc_nbytes=1024 ** 2, rdcc_nslots=10000)
            self.file_handles[str(file_path)] = h5_file

            n_trials = h5_file["embeddings_sequence"].shape[0]
            if max_trials_per_file:
                n_trials = min(n_trials, max_trials_per_file)

            for trial_idx in range(n_trials):
                self.samples.append((str(file_path), trial_idx))

    def __getitem__(self, idx):
        file_path, trial_idx = self.samples[idx]
        h5_file = self.file_handles[file_path]

        seq = torch.from_numpy(h5_file["embeddings_sequence"][trial_idx]).float()

        label_super = -1
        if "super_category_ids" in h5_file and trial_idx < len(h5_file["super_category_ids"]):
            label_super = int(h5_file["super_category_ids"][trial_idx])

            # Convert 1-7 to 0-6 for PyTorch embeddings
            if 1 <= label_super <= 7:
                label_super = label_super - 1  # Now 0-6
            else:
                label_super = -1  # Invalid, filter out

        return seq, torch.tensor(label_super, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __del__(self):
        for h5_file in self.file_handles.values():
            h5_file.close()


class GPUCachedEEGDataset(Dataset):
    """Cache data directly on GPU for zero-copy access"""

    def __init__(self, file_paths, data_loader, max_trials_per_file=10000, device='cuda:0'):
        self.device = device
        self.samples = []

        logger.info(f"Caching data to GPU {device}...")

        for file_path in tqdm(file_paths, desc="Loading to GPU"):
            with h5py.File(file_path, 'r') as h5_file:
                # Load directly to GPU
                eeg_data = torch.from_numpy(h5_file["embeddings_sequence"][:max_trials_per_file]).float().to(device)

                if "super_category_ids" in h5_file:
                    labels = torch.from_numpy(h5_file["super_category_ids"][:max_trials_per_file]).long().to(device)
                    # Fix 1-7 -> 0-6
                    labels = torch.where((labels >= 1) & (labels <= 7), labels - 1, torch.tensor(-1, device=device))
                else:
                    labels = torch.full((eeg_data.shape[0],), -1, dtype=torch.long, device=device)

                # Store indices pointing to GPU tensors
                for i in range(eeg_data.shape[0]):
                    self.samples.append((eeg_data[i], labels[i]))

        logger.info(f"Cached {len(self.samples)} samples on GPU")

    def __getitem__(self, idx):
        # Direct GPU tensor access - zero copy!
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class FastEEGDataset(Dataset):
    """Memory-mapped H5 dataset with minimal overhead"""

    # def __init__(self, file_paths, data_loader, max_trials_per_file=10000):
    #     self.samples = []
    #     self.file_cache = {}
    #
    #     for file_path in file_paths:
    #         # Keep file handles open with memory mapping
    #         h5_file = h5py.File(file_path, 'r', rdcc_nbytes=1024 ** 2 * 100, rdcc_nslots=10007)
    #         self.file_cache[str(file_path)] = h5_file
    #
    #         n_trials = min(h5_file["embeddings_sequence"].shape[0], max_trials_per_file)
    #
    #         for trial_idx in range(n_trials):
    #             self.samples.append((str(file_path), trial_idx))
    def __init__(self, file_paths, data_loader, max_trials_per_file=5000):
        self.samples = []
        self.file_paths = {}  # Store paths, not handles

        for file_path in file_paths:
            self.file_paths[str(file_path)] = file_path

            # Temporarily open to count trials
            with h5py.File(file_path, 'r') as h5_file:
                n_trials = min(h5_file["embeddings_sequence"].shape[0], max_trials_per_file)

                for trial_idx in range(n_trials):
                    self.samples.append((str(file_path), trial_idx))

     def __getitem__(self, idx):
        file_path, trial_idx = self.samples[idx]

        # Open file fresh each time (fast with OS caching)
        with h5py.File(file_path, 'r') as h5_file:
            seq = torch.from_numpy(h5_file["embeddings_sequence"][trial_idx]).float()

            label = -1
            if "super_category_ids" in h5_file and trial_idx < len(h5_file["super_category_ids"]):
                label = int(h5_file["super_category_ids"][trial_idx])
                if 1 <= label <= 7:
                    label = label - 1
                else:
                    label = -1

        return seq, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    # def __del__(self):
    #     for h5_file in self.file_cache.values():
    #         h5_file.close()



from functools import lru_cache


class ShardedPTDataset(Dataset):
    """Dataset with per-worker shard caching"""

    def __init__(self, pt_root, subject_filter=None, trials_per_shard=1000):
        self.pt_root = Path(pt_root)
        self.trials_per_shard = trials_per_shard

        if subject_filter:
            subject_names = [s.name for s in subject_filter]
            self.shard_files = []
            for subject_name in subject_names:
                subject_dir = self.pt_root / subject_name
                if subject_dir.exists():
                    self.shard_files.extend(sorted(subject_dir.glob("*_shard_*.pt")))
        else:
            self.shard_files = sorted(self.pt_root.rglob("*_shard_*.pt"))

        # Build index without loading
        self.samples = []
        for shard_idx, shard_file in enumerate(self.shard_files):
            for local_idx in range(trials_per_shard):
                self.samples.append((shard_idx, local_idx))

        logger.info(f"Indexed {len(self.shard_files)} shards → {len(self.samples)} samples (instant)")

        # Per-worker cache (created lazily in each worker process)
        self._shard_cache = {}

    def _load_shard(self, shard_idx):
        """Load shard with caching - only loads each shard once per worker"""
        if shard_idx not in self._shard_cache:
            shard_file = self.shard_files[shard_idx]
            try:
                self._shard_cache[shard_idx] = torch.load(shard_file, weights_only=True)
            except Exception as e:
                logger.warning(f"Error loading shard {shard_idx}: {e}")
                # Return dummy data
                return {
                    'embeddings': torch.zeros(1000, 301, 1024),
                    'labels': torch.full((1000,), -1, dtype=torch.long)
                }

        return self._shard_cache[shard_idx]

    def __getitem__(self, idx):
        shard_idx, local_idx = self.samples[idx]

        # Load shard (cached after first access)
        data = self._load_shard(shard_idx)

        # Handle edge case where shard has fewer trials
        if local_idx >= len(data['labels']):
            local_idx = len(data['labels']) - 1

        return data['embeddings'][local_idx].float(), data['labels'][local_idx]

    def __len__(self):
        return len(self.samples)


class ShardBatchDataset(Dataset):
    """Load entire shards as batches - no cross-shard mixing"""

    def __init__(self, pt_root, subject_filter=None):
        self.pt_root = Path(pt_root)

        if subject_filter:
            subject_names = [s.name for s in subject_filter]
            self.shard_files = []
            for subject_name in subject_names:
                subject_dir = self.pt_root / subject_name
                if subject_dir.exists():
                    self.shard_files.extend(sorted(subject_dir.glob("*_shard_*.pt")))
        else:
            self.shard_files = sorted(self.pt_root.rglob("*_shard_*.pt"))

        logger.info(f"Found {len(self.shard_files)} shards (each = 1000 trials)")

    def __getitem__(self, idx):
        shard_file = self.shard_files[idx]
        data = torch.load(shard_file)

        # Filter valid labels
        valid_mask = data['labels'] >= 0
        return data['embeddings'][valid_mask], data['labels'][valid_mask].float()

    def __len__(self):
        return len(self.shard_files)


# Custom collate function






if __name__ == "__main__":
    import argparse

    # Argument parser - EXACT from working version with category additions
    parser = argparse.ArgumentParser(description='EEGViTCNet Category-Aware MoE Training')
    # parser.add_argument('--data_path', type=str,
    #                     default="/raid/datasets/tanaya/fm/harmonised_up/",
    #                     help='Path to harmonized EEG data')
    parser.add_argument('--data_paths', nargs='+', type=str,
                        default=["/raid/datasets/tanaya/fm/harmonised_up/"],
                        help='List of paths to harmonized EEG datasets')
    parser.add_argument('--dataset_names', nargs='+', type=str,
                        default=["alljoined"],
                        help='Names for each dataset (e.g., alljoined infant)')
    parser.add_argument('--focus_on_super_categories', action='store_true', default=True,
                        help='Focus on super_categories instead of image_categories')

    parser.add_argument('--output_dir', type=str,
                        default="/raid/datasets/tanaya/fm/eegvit_Results/",
                        help='Output directory for results')
    parser.add_argument('--pt_data_dir', type=str,
                        default="/tmp/tanaya/alljoined/",

                        help='Path to sharded PT files directory')

    parser.add_argument('--train_subjects', type=str, default=None,
                        help='Specific subjects for training (e.g., "1,2,3" or "1-10")')
    parser.add_argument('--val_subjects', type=str, default=None,
                        help='Specific subjects for validation (e.g., "11,12,13")')
    parser.add_argument('--auto_split', action='store_true',
                        help='Automatically split subjects 80/20 for train/val')
    parser.add_argument('--train_dataset', type=str, default=None,
                        choices=['alljoined', 'infant'],
                        help='Specific dataset for training')
    parser.add_argument('--val_dataset', type=str, default=None,
                        choices=['alljoined', 'infant'],
                        help='Specific dataset for validation')

    parser.add_argument('--split_strategy', type=str, default='mixed',
                        choices=['dataset_split', 'mixed', 'leave_one_out'],
                        help='Splitting strategy across datasets')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (optimized for V100 memory)')
    parser.add_argument('--num_experts', type=int, default=8,
                        help='Number of experts in MoE')
    parser.add_argument('--dinov3_size', type=str, default='base',
                        choices=['base', 'large', 'giant'],
                        help='DinoV3 model size for feature alignment')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate for training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test_run', action='store_true',
                        help='Run quick test with minimal data')


    args = parser.parse_args()



    # Handle multi-dataset setup
    if len(args.data_paths) != len(args.dataset_names):
        logger.warning("Number of data paths and dataset names don't match. Auto-generating names.")
        args.dataset_names = [f"dataset_{i}" for i in range(len(args.data_paths))]

    logger.info(f"Training on datasets:")
    for name, path in zip(args.dataset_names, args.data_paths):
        logger.info(f"  {name}: {path}")


    try:
        #Create enhanced data loader
        data_loader = EEGViTDataLoader(
            harmonized_dirs=args.data_paths,
            dataset_names=args.dataset_names,
            use_super_categories=args.focus_on_super_categories,
            batch_size=args.batch_size,
            max_trials_per_subject=10000  # LIMIT TO 5000 TRIALS PER SUBJECT, now don't matter because I am using all 83k trials with shards..
        )


        # Run main training pipeline
        model, trainer = main_training_pipeline(
            data_loader=data_loader,
            output_dir=str(args.output_dir),
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            num_experts=args.num_experts,
            dinov3_size=args.dinov3_size
        )

           # Get all subject directories from PT shard location
        pt_root = Path(args.pt_data_dir)  # e.g., /home/tanaya/phase2_2/alljoined
        all_subject_dirs = sorted([d for d in pt_root.iterdir() if d.is_dir() and d.name.startswith('sub-')])

        logger.info(f"Found {len(all_subject_dirs)} subject directories in {pt_root}")

        # Split subjects (not files) for train/val
        if args.auto_split:
            # 80/20 split by subject
            train_size = int(len(all_subject_dirs) * 0.8)
            train_subject_dirs = all_subject_dirs[:train_size]
            val_subject_dirs = all_subject_dirs[train_size:]
        else:
            # Manual subject selection
            train_subject_ids = parse_subject_list(args.train_subjects) if args.train_subjects else []
            val_subject_ids = parse_subject_list(args.val_subjects) if args.val_subjects else []

            train_subject_dirs = [d for d in all_subject_dirs if d.name in train_subject_ids]
            val_subject_dirs = [d for d in all_subject_dirs if d.name in val_subject_ids]

        logger.info(f"Split: {len(train_subject_dirs)} train subjects, {len(val_subject_dirs)} val subjects")

        # Create datasets from subject directories
        # train_dataset = ShardedPTDataset(pt_root, subject_filter=train_subject_dirs)
        # val_dataset = ShardedPTDataset(pt_root, subject_filter=val_subject_dirs) if val_subject_dirs else None
        train_dataset = ShardBatchDataset(pt_root, subject_filter=train_subject_dirs)
        val_dataset = ShardBatchDataset(pt_root, subject_filter=val_subject_dirs) if val_subject_dirs else None


        def shard_collate_fn(batch):
            # batch is list of (embeddings, labels) from different shards
            # Flatten into single batch
            all_emb = torch.cat([item[0] for item in batch], dim=0)
            all_labels = torch.cat([item[1] for item in batch], dim=0)
            return all_emb, all_labels
            # Training loop continues as before...
        # Enhanced training loop with super-category evaluation


         # DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,  # 1 shard at a time
            shuffle=True,  # Shuffle shard order
            num_workers=4,  # Fewer workers
            collate_fn=shard_collate_fn
        )

        # val_loader = DataLoader(
        #     val_dataset,
        #     batch_size=args.batch_size,
        #     shuffle=False,
        #     num_workers=8,
        #     pin_memory=True
        # )

        val_loader = DataLoader(
            # train_dataset,
            val_dataset,
            batch_size=1,  # 1 shard at a time
            shuffle=False,  # Shuffle shard order
            num_workers=2,  # Fewer workers
            collate_fn=shard_collate_fn

        )if val_dataset else None
   
        best_loss = float('inf')
        best_super_accuracy = 0.0
        best_val_loss = float('inf')

        from tqdm import tqdm
        import time
       
        for epoch in range(args.epochs):
            trainer.anneal_moe_temperature(epoch)
            model.train()
            total_loss = 0
            total_super_loss = 0
            total_cb_focal_loss = 0
            num_batches = 0
            expert_usage_sum = torch.zeros(args.num_experts)

            # ADding Accuracy tracking
            train_correct = 0
            train_total = 0

            # Per-category tracking (OUTSIDE inner loop)
            correct_per_category = defaultdict(int)
            total_per_category = defaultdict(int)


            epoch_start = time.time()

            # Add tqdm progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", total=len(train_loader))

            for batch_idx, (seq_emb, labels_super) in enumerate(pbar):
                batch_start = time.time()
                # Convert dtype FIRST
                seq_emb = seq_emb.float()  # float16 → float32
                labels_super = labels_super.long()
                # THEN move to device
                seq_emb = seq_emb.to(trainer.device)
                labels_super = labels_super.to(trainer.device)

                # Filter valid samples
                valid_mask = labels_super >= 0
                if valid_mask.sum() == 0:
                    continue

                seq_emb = seq_emb[valid_mask]
                labels_super = labels_super[valid_mask]

                # Split into model-sized batches (512 trials at a time)
                for i in range(0, len(seq_emb), 512):
                    batch_seq = seq_emb[i:i + 512]
                    batch_labels = labels_super[i:i + 512]

                    outputs = model(batch_seq, super_category_ids=batch_labels)
                    loss_dict = trainer.compute_loss(outputs, super_category_ids=batch_labels)



                    # ADding  accuracy during training
                    with torch.no_grad():
                        preds = torch.argmax(outputs['predicted_super_categories'], dim=1)
                        # correct = (preds == batch_labels).sum().item()

                        # correct_predictions += correct
                        correct_mask = (preds == batch_labels)
                        # Overall accuracy
                        train_correct += correct_mask.sum().item()
                        train_total += len(batch_labels)
                        # Per-category accuracy
                        for pred, label in zip(correct_mask.cpu().numpy(), batch_labels.cpu().numpy()):
                            total_per_category[label] += 1
                            if pred:
                                 correct_per_category[label] += 1

                        # # Track category distribution
                        # for label in batch_labels.cpu().numpy():
                        #     category_distribution[label] += 1

                    trainer.optimizer.zero_grad()
                    loss_dict['total_loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    trainer.optimizer.step()

                    total_loss += loss_dict['total_loss'].item()
                    total_super_loss += loss_dict.get('super_category_loss', 0).item()
                    total_cb_focal_loss += loss_dict.get('cb_focal_super_loss', 0).item()
                    num_batches += 1

                    # Accumulate expert usage
                    if 'expert_usage' in outputs:
                        expert_usage_sum += outputs['expert_usage'].cpu()

                batch_time = time.time() - batch_start

                # Update progress bar with accuracy
                train_accuracy = train_correct / max(train_total, 1) * 100

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss'].item():.4f}",
                    'cb_focal': f"{loss_dict.get('cb_focal_super_loss', 0).item():.4f}",
                    'acc': f"{train_accuracy:.1f}%",
                    'time': f"{batch_time:.2f}s"
                })

                # Detailed logging every 100 batches
                if batch_idx % 100 == 0:
                    moe = trainer.model.module.moe if hasattr(trainer.model, 'module') else trainer.model.moe
                    current_temp = moe.temperature.item()

                    # if torch.cuda.is_available():
                    #     gpu_memory = [torch.cuda.memory_allocated(i) / 1024 ** 3
                    #                   for i in range(torch.cuda.device_count())]
                    logger.info(
                        f"Epoch {epoch} Batch {batch_idx}: "
                        f"Loss={loss_dict['total_loss'].item():.4f}, "
                        f"CB-Focal={loss_dict.get('cb_focal_super_loss', 0).item():.4f}, "
                        f"Category Accuracy={train_accuracy:.1f}%, "
                        f"Temp={current_temp:.3f}, "
                        # f"GPU Memory: {[f'{m:.1f}GB' for m in gpu_memory]}"
                    )

            #
            pbar.close()

            # VALIDATION ACCURACY (fast, uses val_loader)
            val_correct = 0
            val_total = 0
            val_total_loss = 0
            val_num_batches = 0

            if val_loader:
                model.eval()
                with torch.no_grad():
                    for seq_emb, labels_super in val_loader:
                        seq_emb = seq_emb.float().to(trainer.device)
                        labels_super = labels_super.long().to(trainer.device)

                        valid_mask = labels_super >= 0
                        if valid_mask.sum() == 0:
                            continue

                        seq_emb = seq_emb[valid_mask]
                        labels_super = labels_super[valid_mask]

                        # Process in chunks
                        for i in range(0, len(seq_emb), 512):
                            batch_seq = seq_emb[i:i + 512]
                            batch_labels = labels_super[i:i + 512]

                            outputs = model(batch_seq, super_category_ids=batch_labels)
                            loss_dict = trainer.compute_loss(outputs, super_category_ids=batch_labels)

                            # Accumulate validation loss
                            val_total_loss += loss_dict['total_loss'].item()
                            val_num_batches += 1

                            preds = torch.argmax(outputs['predicted_super_categories'], dim=1)

                            val_correct += (preds == batch_labels).sum().item()
                            val_total += len(batch_labels)

                model.train()

            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_time / 60:.1f} minutes")

            # After epoch ends, get final temperature for summary
            moe = trainer.model.module.moe if hasattr(trainer.model, 'module') else trainer.model.moe
            current_temp = moe.temperature.item()

            avg_train_loss = total_loss / max(num_batches, 1)
            # avg_loss = total_loss / max(num_batches, 1)
            avg_super_loss = total_super_loss / max(num_batches, 1)
            avg_cb_focal = total_cb_focal_loss / max(num_batches, 1)
            avg_expert_usage = expert_usage_sum / max(num_batches, 1)
            train_accuracy = train_correct / max(train_total, 1) * 100
            avg_val_loss = val_total_loss / max(val_num_batches, 1) if val_num_batches > 0 else float('inf')
            val_accuracy = val_correct / max(val_total, 1) * 100 if val_total > 0 else 0

            # Epoch summary with validation metrics
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Epoch {epoch} Summary ({epoch_time / 60:.1f} min):")
            logger.info(f"  Training:")
            logger.info(f"    Loss: {avg_train_loss:.4f}")
            logger.info(f"    Accuracy: {train_accuracy:.1f}% ({train_correct}/{train_total})")

            # logger.info(f"\n{'=' * 60}")
            # logger.info(f"Epoch {epoch} Summary:")
            # logger.info(f"  Batches processed: {num_batches}")
            # logger.info(f"  Avg Total Loss: {avg_loss:.4f}")
            # logger.info(f"  Avg Super Category Loss: {avg_super_loss:.4f}")
            # logger.info(f"  Avg CB-Focal Loss: {avg_cb_focal:.4f}")
            # # logger.info(f"  Training Accuracy: {train_accuracy:.1f}%")  # NEW
            # logger.info(f"  Train Accuracy: {train_accuracy:.1f}% ({train_correct}/{train_total})")
            if val_total > 0:
                logger.info(f"  Validation:")
                logger.info(f"   Val Loss: {avg_val_loss:.4f}")
                logger.info(f"  Val Accuracy: {val_accuracy:.1f}% ({val_correct}/{val_total})")

            # logger.info(f"  Samples: {total_predictions}")
            logger.info(f"  MoE Temperature: {current_temp:.4f}")
            logger.info(f"  Expert Usage: {avg_expert_usage.numpy()}")
            logger.info(f"  Expert Std Dev: {avg_expert_usage.std():.4f}")
            logger.info(f"{'=' * 60}\n")

            # Save best model on training loss
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                checkpoint_path = Path(args.output_dir) / f"best_train_loss_model_epoch_{epoch}.pt"
                save_enhanced_checkpoint(model, trainer, data_loader, checkpoint_path)
                logger.info(f"Saved best model with TRAINING loss: {avg_train_loss:.4f}")

            # Save best model based on VALIDATION LOSS (separate tracking)
            if val_num_batches > 0  and avg_val_loss < best_val_loss:  # SIMPLIFIED
                    best_val_loss = avg_val_loss
                    val_loss_path = Path(args.output_dir) / f"best_val_loss_epoch_{epoch}.pt"
                    save_enhanced_checkpoint(model, trainer, data_loader, val_loss_path)
                    logger.info(f"✓ Saved best validation loss model: {avg_val_loss:.4f}")

            # Save best model based on VALIDATION ACCURACY (separate check)
            if val_total > 0 and val_accuracy > best_super_accuracy:
                best_super_accuracy = val_accuracy
                best_acc_path = Path(args.output_dir) / f"best_val_accuracy_model_epoch_{epoch}.pt"
                save_enhanced_checkpoint(model, trainer, data_loader, best_acc_path)
                logger.info(f"New best validation accuracy: {val_accuracy:.1f}%")

                if val_accuracy > 60:
                    logger.info("Model ready for next stage (>60% accuracy)")


        try:
            analysis = model.analyze_expert_specialization()
            logger.info("Final Expert Analysis:")
            logger.info(f"  Expert Usage: {analysis['expert_usage'].cpu().numpy()}")
            logger.info(f"  Routing Temperature: {analysis['routing_temperature'].item():.3f}")



        except Exception as e:
            logger.warning(f"Final analysis failed: {e}")

        # Save final model
        final_model_path = Path(args.output_dir) / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model.config.__dict__,
            'training_complete': True,
            'args': vars(args)
        }, final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

        logger.info("All outputs saved successfully!")


        logger.info("Training completed successfully!")



    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()

    