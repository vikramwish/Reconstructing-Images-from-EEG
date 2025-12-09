# Set torch compilation settings

import os
import torch._dynamo
#os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU-only mode

torch._dynamo.config.suppress_errors = True
os.environ["TORCH_LOGS"] = ""

os.environ['TORCHDYNAMO_DISABLE'] = "1"
# optional: ensure no weird TORCH_LOGS is present
if 'TORCH_LOGS' in os.environ:
    del os.environ['TORCH_LOGS']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.cuda.amp import autocast, GradScaler
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import math
from dataclasses import dataclass
import h5py
from pathlib import Path

import json
from tqdm import tqdm
from datetime import datetime
import time
import argparse
import mne
from mne.channels import make_standard_montage
import pandas as pd
# Suppress torch warnings
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
warnings.filterwarnings("ignore", message="torch.cuda.amp.autocast")



@dataclass
class HarmonizationConfig:
    """Configuration for LCM + Temporal Transformer"""
    # Channel mapping parameters -  for spatial awareness
    max_input_channels: int = 128
    virtual_channels: int = 128  # Increased to support spatial interpolation
    channel_embedding_dim: int = 256
    anatomical_channels: int = 94  # Standard 10-20 extended
    enable_two_stage: bool = True  #

    # Temporal transformer parameters
    max_time_points: int = 512  # Max time points across all datasets
    time_embedding_dim: int = 256
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 6

    # Output parameters - FIXED to 1024
    output_embedding_dim: int = 1024  # Fixed embedding dimension

    # Training parameters
    dropout: float = 0.1
    use_positional_encoding: bool = True

    # Spatial parameters - NEW
    use_spatial_embeddings: bool = True
    spatial_embedding_dim: int = 64
    spatial_dim: int = 3  # x,y,z coordinates
    target_montage: str = "standard_1020"  # Reference montage for harmonisation

    # Montage information
    montage_types: List[str] = None

    def __post_init__(self):
        if self.montage_types is None:
            self.montage_types = ["10-20", "10-10", "10-5", "custom"]


class MNEMontageHandler:
    """MNE-based electrode position handler for anatomical consistency"""

    def __init__(self, target_montage: str = "standard_1020"):
        self.target_montage = target_montage
        try:
            self.montage = make_standard_montage(target_montage)

            # Extract positions - MNE stores positions differently
            self.montage_positions = {}

            # Get channel positions directly from montage
            pos = self.montage.get_positions()
            if 'ch_pos' in pos:
                for ch_name, position in pos['ch_pos'].items():
                    # Position is in meters, convert to more readable scale
                    self.montage_positions[ch_name] = (position * 100).tolist()  # Convert to cm

            print(f"Loaded {len(self.montage_positions)} electrode positions from {target_montage}")
            print(f"Available channels: {list(self.montage_positions.keys())[:10]}...")

        except Exception as e:
            print(f"Error loading montage {target_montage}: {e}")
            # Fallback to empty positions
            self.montage_positions = {}

    def get_position(self, channel_name: str) -> Optional[List[float]]:
        """Get 3D position for a channel name from MNE montage"""
        # Clean channel name
        clean_name = channel_name.strip().replace(' ', '')

        # Direct lookup in montage
        if clean_name in self.montage_positions:
            return self.montage_positions[clean_name]

        # Case-insensitive fallback
        for ch_name, pos in self.montage_positions.items():
            if ch_name.lower() == clean_name.lower():
                return pos

        # Alternative naming conventions
        name_mappings = {
            'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
            'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'
        }

        if clean_name in name_mappings:
            alt_name = name_mappings[clean_name]
            if alt_name in self.montage_positions:
                return self.montage_positions[alt_name]

        return None

    def infer_montage_type(self, channel_names: List[str]) -> Tuple[str, float]:
        """Infer source montage type using MNE montages"""
        channels_set = set(ch.strip() for ch in channel_names)

        # Try different standard montages
        test_montages = ["standard_1020", "standard_1005", "biosemi64", "easycap-M1"]
        best_match = "custom"
        best_score = 0

        for montage_name in test_montages:
            try:
                test_montage = make_standard_montage(montage_name)
                pos = test_montage.get_positions()
                if 'ch_pos' in pos:
                    montage_channels = set(pos['ch_pos'].keys())
                    overlap = len(channels_set.intersection(montage_channels))
                    score = overlap / len(channels_set) if len(channels_set) > 0 else 0

                    if score > best_score:
                        best_match = montage_name
                        best_score = score

            except Exception:
                continue

        return best_match, best_score

    def map_channels_to_positions(self, channel_names: List[str]) -> torch.Tensor:
        """Map channel names to 3D positions using MNE montage"""
        positions = []
        missing_channels = []
        found_channels = []

        for name in channel_names:
            pos = self.get_position(name)
            if pos is None:
                # Default position for unknown channels (origin)
                pos = [0.0, 0.0, 0.0]
                missing_channels.append(name)
            else:
                found_channels.append(name)
            positions.append(pos)

        # Only warn if many channels are missing
        if len(missing_channels) > len(found_channels):
            print(f"Warning: {len(missing_channels)} channels not found in {self.target_montage}")
            print(f"Missing: {missing_channels[:10]}...")
            print(f"Found: {found_channels[:10]}...")
            print(f"Available in montage: {list(self.montage_positions.keys())[:10]}...")

        return torch.tensor(positions, dtype=torch.float32)


def debug_channel_mapping(data_dir: Path, target_montage: str = "standard_1020"):
    """Debug function to check channel mapping issues"""

    print(f"=== DEBUGGING CHANNEL MAPPING ===")
    print(f"Target montage: {target_montage}")

    # Load MNE montage
    try:
        montage = make_standard_montage(target_montage)
        pos = montage.get_positions()
        available_channels = list(pos['ch_pos'].keys()) if 'ch_pos' in pos else []

        print(f"Available channels in {target_montage}: {len(available_channels)}")
        print(f"Sample channels: {sorted(available_channels)[:20]}")
    except Exception as e:
        print(f"Error loading {target_montage}: {e}")
        return

    # Check your data
    loader = EEGDataLoader(data_dir, batch_size=1)
    if len(loader.subject_files) > 0:
        sample_file = loader.subject_files[0]
        metadata = loader.metadata_cache.get(str(sample_file), {})
        your_channels = metadata.get('channel_names', [])

        print(f"\nYour channels ({len(your_channels)}): {your_channels}")

        # Check overlap
        available_set = set(available_channels)
        your_set = set(your_channels)
        matching = your_set.intersection(available_set)
        missing = your_set - available_set

        print(f"\nMatching channels ({len(matching)}): {sorted(list(matching))}")
        print(f"Missing channels ({len(missing)}): {sorted(list(missing))}")

        coverage = len(matching) / len(your_channels) if your_channels else 0
        print(f"Coverage: {coverage:.2%}")

        # Test other montages for better coverage
        print(f"\nTesting coverage with other montages:")
        for test_montage in ["standard_1005", "biosemi64", "easycap-M1", "standard_1020"]:
            try:
                test_mont = make_standard_montage(test_montage)
                test_pos = test_mont.get_positions()
                if 'ch_pos' in test_pos:
                    test_channels = set(test_pos['ch_pos'].keys())
                    overlap = len(your_set.intersection(test_channels))
                    test_coverage = overlap / len(your_channels)
                    print(f"  {test_montage}: {overlap}/{len(your_channels)} channels ({test_coverage:.2%})")
            except:
                print(f"  {test_montage}: Error loading")

class ChannelEmbedding(nn.Module):
    """Enhanced channel embeddings with MNE-based spatial awareness"""

    def __init__(self, config: HarmonizationConfig):
        super().__init__()
        self.config = config
        self.use_spatial = config.use_spatial_embeddings

        # Create embedding table for all possible channels
        self.channel_embeddings = nn.Embedding(
            config.max_input_channels,
            config.channel_embedding_dim
        )

        # Montage-specific encodings
        self.montage_embeddings = nn.ModuleDict({
            montage: nn.Embedding(1, config.channel_embedding_dim)
            for montage in config.montage_types
        })

        # Enhanced spatial embeddings with configurable input dimension
        if self.use_spatial:
            self.spatial_embeddings = nn.Sequential(
                nn.Linear(config.spatial_dim, config.spatial_embedding_dim),  # Uses config.spatial_dim
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.spatial_embedding_dim, config.channel_embedding_dim)
            )

            # Spatial attention for position-aware channel selection
            self.spatial_attention = nn.MultiheadAttention(
                embed_dim=config.channel_embedding_dim,
                num_heads=4,
                dropout=config.dropout,
                batch_first=True
            )

        # Initialize embeddings
        nn.init.normal_(self.channel_embeddings.weight, std=0.02)

    def forward(
            self,
            channel_indices: torch.Tensor,
            montage_type: Optional[str] = None,
            channel_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Base channel embeddings
        embeddings = self.channel_embeddings(channel_indices)

        # Add montage-specific information
        if montage_type and montage_type in self.montage_embeddings:
            montage_emb = self.montage_embeddings[montage_type](
                torch.zeros(1, dtype=torch.long, device=channel_indices.device)
            )
            embeddings = embeddings + montage_emb.unsqueeze(0)

        # Enhanced spatial position handling
        if self.use_spatial and channel_positions is not None:
            spatial_emb = self.spatial_embeddings(channel_positions)

            # Apply spatial attention to make embeddings position-aware
            attended_spatial, _ = self.spatial_attention(
                embeddings, spatial_emb, spatial_emb
            )
            embeddings = embeddings + attended_spatial

        return embeddings


class TwoStageLearnedChannelMapping(nn.Module):
    """Two-stage harmonization: Input → Anatomical (94) → Virtual (128)"""

    def __init__(self, config: HarmonizationConfig):
        super().__init__()
        self.config = config
        self.anatomical_channels = 94  # Fixed for standard_10_20_ext

        # Stage 1: Input → Anatomical mapping
        self.channel_embedding = ChannelEmbedding(config)

        self.anatomical_attention = nn.MultiheadAttention(
            embed_dim=config.channel_embedding_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )

        self.anatomical_queries = nn.Parameter(
            torch.randn(self.anatomical_channels, config.channel_embedding_dim)
        )

        # Stage 2: Anatomical → Virtual expansion
        self.anatomical_to_embed = nn.Linear(
            self.anatomical_channels, config.channel_embedding_dim
        )

        self.virtual_queries = nn.Parameter(
            torch.randn(config.virtual_channels, config.channel_embedding_dim)
        )

        self.virtual_attention = nn.MultiheadAttention(
            embed_dim=config.channel_embedding_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )

        # Normalization layers
        self.anatomical_norm = nn.LayerNorm(config.channel_embedding_dim)
        self.virtual_norm = nn.LayerNorm(config.channel_embedding_dim)

    def forward(
            self,
            eeg_data: torch.Tensor,
            channel_mask: Optional[torch.Tensor] = None,
            channel_indices: Optional[torch.Tensor] = None,
            montage_type: Optional[str] = None,
            channel_positions: Optional[torch.Tensor] = None,
            return_virtual_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, n_channels, time_points = eeg_data.shape

        # Stage 1: Map to anatomical space
        anatomical_eeg, anatomical_attn = self._map_to_anatomical_space(
            eeg_data, channel_mask, channel_indices, montage_type, channel_positions
        )

        # Stage 2: Expand to virtual space
        virtual_eeg, virtual_attn = self._expand_to_virtual_space(anatomical_eeg)

        if return_virtual_attention:
            return virtual_eeg, (anatomical_attn, virtual_attn)
        else:
            return virtual_eeg, anatomical_attn

    def _map_to_anatomical_space(self, eeg_data, channel_mask, channel_indices, montage_type, channel_positions):
        batch_size, n_channels, time_points = eeg_data.shape

        # Get channel embeddings
        channel_emb = self.channel_embedding(
            channel_indices, montage_type, channel_positions
        )

        # Prepare anatomical queries
        anatomical_queries = self.anatomical_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Attention mask
        key_padding_mask = None
        if channel_mask is not None:
            key_padding_mask = ~channel_mask

        # Anatomical attention
        anatomical_features, anatomical_attn = self.anatomical_attention(
            anatomical_queries, channel_emb, channel_emb,
            key_padding_mask=key_padding_mask
        )

        anatomical_features = self.anatomical_norm(anatomical_features + anatomical_queries)

        # Apply to EEG signals
        # eeg_transposed = eeg_data.transpose(1, 2)  # [batch, time, channels]
        # anatomical_eeg = torch.bmm(anatomical_attn, eeg_transposed)  # [batch, 94, time]
        anatomical_eeg = torch.bmm(anatomical_attn, eeg_data)
        anatomical_eeg = anatomical_eeg.transpose(1, 2)  # [batch, time, 94]

        return anatomical_eeg, anatomical_attn

    def _expand_to_virtual_space(self, anatomical_eeg):
        batch_size, n_time, n_anatomical = anatomical_eeg.shape

        # CRITICAL: Preserve temporal dynamics with linear projection
        # anatomical_embeddings = self.anatomical_to_embed(
        #     anatomical_eeg.transpose(1, 2)  # [batch, 94, time]
        # ).transpose(1, 2)  # [batch, time, embed_dim]
        anatomical_embeddings = self.anatomical_to_embed(anatomical_eeg)

        # Temporal pooling for attention (preserving time info)
        anatomical_features = torch.mean(anatomical_embeddings, dim=1, keepdim=True).expand(
            -1, n_anatomical, self.config.channel_embedding_dim
        )

        # Virtual queries
        virtual_queries = self.virtual_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Virtual attention
        virtual_features, virtual_attn = self.virtual_attention(
            virtual_queries, anatomical_features, anatomical_features
        )

        virtual_features = self.virtual_norm(virtual_features + virtual_queries)

        # Apply to anatomical signals
        anatomical_eeg_transposed = anatomical_eeg.transpose(1, 2)  # [batch, 94, time]
        virtual_eeg = torch.bmm(virtual_attn, anatomical_eeg_transposed)  # [batch, 128, time]

        return virtual_eeg, virtual_attn

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal dimension"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """Enhanced transformer that outputs both pooled and sequence embeddings"""

    def __init__(self, config: HarmonizationConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.virtual_channels, config.transformer_dim)

        # Separate projections for different output types
        self.cls_proj = nn.Linear(config.transformer_dim, config.output_embedding_dim)
        self.sequence_proj = nn.Linear(config.transformer_dim, config.output_embedding_dim)

        if config.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                config.transformer_dim,
                config.max_time_points,
                config.dropout
            )

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.transformer_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers
        )

    def forward(self, eeg_data, time_mask=None, output_type='both'):
        """
        Args:
            output_type: 'pooled', 'sequence', or 'both'
        """
        batch_size, virtual_channels, time_points = eeg_data.shape

        # Transpose and project: [B, T, V] -> [B, T, D]
        x = eeg_data.transpose(1, 2)
        x = self.input_proj(x)

        # Add positional encoding
        if self.config.use_positional_encoding:
            x = self.pos_encoder(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Handle time mask
        src_key_padding_mask = None
        if time_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=time_mask.device, dtype=time_mask.dtype)
            extended_mask = torch.cat([cls_mask, time_mask], dim=1)
            src_key_padding_mask = ~extended_mask

        # Apply transformer
        transformed = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        outputs = {}

        if output_type in ['pooled', 'both']:
            # CLS token representation (pooled)
            cls_output = transformed[:, 0, :]
            outputs['pooled'] = self.cls_proj(cls_output)

        if output_type in ['sequence', 'both']:
            # Sequence features (without CLS token)
            sequence_features = transformed[:, 1:, :]  # [B, T, D]
            batch_size, seq_len, dim = sequence_features.shape
            sequence_flat = sequence_features.reshape(-1, dim)
            projected_flat = self.sequence_proj(sequence_flat)
            outputs['sequence'] = projected_flat.reshape(batch_size, seq_len, self.config.output_embedding_dim)

        return outputs


class EEGHarmonizationLayer(nn.Module):
    """Enhanced harmonization layer with MNE-based spatial awareness"""

    def __init__(self, config: HarmonizationConfig):
        super().__init__()
        self.config = config
        self.montage_handler = MNEMontageHandler(config.target_montage)  # Use MNE handler

        # self.lcm = LearnedChannelMapping(config)
        self.lcm = TwoStageLearnedChannelMapping(config)
        self.temporal_transformer = TemporalTransformer(config)

        self.channel_norm = nn.BatchNorm1d(config.virtual_channels, affine=True)
        self.sampling_rate_embedding = nn.Embedding(10, config.virtual_channels)

    def forward(
            self,
            eeg_data: torch.Tensor,
            channel_mask: Optional[torch.Tensor] = None,
            time_mask: Optional[torch.Tensor] = None,
            channel_indices: Optional[torch.Tensor] = None,
            channel_names: Optional[List[str]] = None,
            montage_type: Optional[str] = None,
            sampling_rate_id: Optional[int] = None,
            output_type: str = 'both',
            return_attention: bool = False,
            return_virtual_attention: bool = False  # NEW PARAMETER
    ) -> Dict[str, torch.Tensor]:

        batch_size = eeg_data.shape[0]

        # Get MNE-based channel positions if channel names provided
        channel_positions = None
        inferred_montage = None

        if channel_names is not None and self.config.use_spatial_embeddings:
            channel_positions = self.montage_handler.map_channels_to_positions(channel_names)
            channel_positions = channel_positions.to(eeg_data.device)

            # Infer source montage for logging
            inferred_montage, match_score = self.montage_handler.infer_montage_type(channel_names)

            # Expand for batch if needed
            if channel_positions.dim() == 2:
                channel_positions = channel_positions.unsqueeze(0).expand(batch_size, -1, -1)

        # Use inferred montage if not provided
        if montage_type is None:
            montage_type = inferred_montage

        # Rest of the forward method remains the same...
        # Step 1: Spatial-aware channel mapping (Enhanced LCM)
        # mapped_eeg, channel_attention = self.lcm(
        #     eeg_data, channel_mask, channel_indices, montage_type, channel_positions
        # )
        mapped_eeg, attention_info = self.lcm(
            eeg_data, channel_mask, channel_indices, montage_type, channel_positions,
            return_virtual_attention=return_virtual_attention
        )
        # Apply channel normalization
        mapped_eeg = self.channel_norm(mapped_eeg)

        if return_virtual_attention:
            anatomical_attention, virtual_attention = attention_info
        else:
            anatomical_attention = attention_info
            virtual_attention = None

        # Apply sampling rate scaling if provided
        if sampling_rate_id is not None:
            rate_scale = self.sampling_rate_embedding(
                torch.tensor([sampling_rate_id], device=eeg_data.device)
            ).squeeze(0)
            mapped_eeg = mapped_eeg * rate_scale.unsqueeze(0).unsqueeze(-1)

        # Step 2: Enhanced temporal transformer
        transformer_outputs = self.temporal_transformer(
            mapped_eeg, time_mask, output_type=output_type
        )

        # Prepare output
        output = {}
        if 'pooled' in transformer_outputs:
            output['embedding_pooled'] = transformer_outputs['pooled']
        if 'sequence' in transformer_outputs:
            output['embedding_sequence'] = transformer_outputs['sequence']

        # if return_attention:
        #     output["channel_attention"] = channel_attention
        #     output["inferred_montage"] = inferred_montage

        if return_attention:
            output["anatomical_attention"] = anatomical_attention
            output["inferred_montage"] = inferred_montage
            if return_virtual_attention:
                output["virtual_attention"] = virtual_attention

        del mapped_eeg
        return output



# 1. new EEGDataLoader class to load image category information
class EEGDataLoader:
    """Enhanced memory-efficient data loader for HDF5 EEG files with image categories"""

    def __init__(
            self,
            data_dir: Path,
            batch_size: int = 32,
            max_channels: int = 128,
            channel_mapping: Optional[Dict[str, int]] = None
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_channels = max_channels
        self.channel_mapping = channel_mapping or {}

        # Scan for all subject files
        self.subject_files = sorted(data_dir.glob("*/sub-*_processed.h5"))
        self.metadata_cache = {}
        self.image_category_cache = {}  # Cache for image categories

        # Load metadata and image categories from files
        self._load_metadata_cache()
        self._load_image_categories_cache()  #  Load image category information

        print(f"Found {len(self.subject_files)} files in {data_dir}")
        print(f"Loaded image categories for {len(self.image_category_cache)} subjects")

    def _load_image_categories_cache(self):
        """Load image category information from mapping_trial.parquet files"""
        import pandas as pd

        for file_path in self.subject_files:
            try:
                # Look for mapping_trial.parquet in the same directory
                parquet_file = file_path.parent / "mapping_trial.parquet"
                if parquet_file.exists():
                    image_mapping = pd.read_parquet(parquet_file)

                    # FILTER to only keep the columns I need
                    required_columns = [
                        'fname', 'category_name', 'category_num', 'category_img_num',
                        'super_category', 'minor_category', 'super_category_id',
                        'minor_category_id', 'image_path_full', 'event_code', 'event_time'
                    ]

                    # Keep only required columns that exist
                    available_columns = [col for col in required_columns if col in image_mapping.columns]
                    image_mapping = image_mapping[available_columns]

                    self.image_category_cache[str(file_path)] = image_mapping
                    print(f"  Loaded image categories for {file_path.name}: {len(image_mapping)} trials")
                else:
                    print(f"  Warning: No image categories found for {file_path.name}")
                    # CHANGE the expected columns in the empty DataFrame
                    self.image_category_cache[str(file_path)] = pd.DataFrame(columns=[
                        'fname', 'category_name', 'category_num', 'category_img_num',
                        'super_category', 'minor_category', 'super_category_id',
                        'minor_category_id', 'image_path_full', 'event_code', 'event_time'
                    ])
            except Exception as e:
                print(f"Error loading image categories for {file_path}: {e}")
                self.image_category_cache[str(file_path)] = pd.DataFrame()

    def load_subject_with_categories(self, file_path: Path) -> Dict:
        """Load subject data with conditional category fields"""

        with h5py.File(file_path, 'r') as f:
            # Load embeddings
            pooled_emb = torch.from_numpy(f['embeddings_pooled'][:])
            sequence_emb = torch.from_numpy(f['embeddings_sequence'][:])

            # Change the categories dictionary field names:
            categories = {
                # Changing 'image_categories' -> use 'category_name'
                'image_categories': [cat.decode('utf-8') for cat in f['category_name'][:]],
                'super_categories': [cat.decode('utf-8') for cat in f['super_categories'][:]],  # KEEP SAME
                'super_category_ids': f['super_category_ids'][:].tolist(),  # KEEP SAME

                # Adding new category fields:
                'minor_categories': [cat.decode('utf-8') for cat in f['minor_category'][:]],
                'minor_category_ids': f['minor_category_ids'][:].tolist(),
                'category_nums': f['category_num'][:].tolist(),
                'category_img_nums': f['category_img_num'][:].tolist(),

                # Changing  use 'fname' for filenames
                'image_filenames': [fname.decode('utf-8') for fname in f['fname'][:]],
                # 'mapped_flags': f['mapped'][:].tolist(),  # KEEP SAME
            }
            #  Load animate/inanimate only if present
            if f.attrs.get('has_sub_category', False):
                categories['sub_category'] = [cat.decode('utf-8') for cat in f['sub_category'][:]]
            else:
                categories['sub_category'] = None  # Explicitly None for AllJoined

            return {
                'embeddings_pooled': pooled_emb,
                'embeddings_sequence': sequence_emb,
                'categories': categories,
                'metadata': dict(f.attrs)
            }

    def _stratified_trial_selection(
            self,
            data: torch.Tensor,
            events: torch.Tensor,
            max_trials: int
    ) -> np.ndarray:
        """Select trials using stratified sampling across event types"""

        if data.shape[0] <= max_trials:
            return np.arange(data.shape[0])

        unique_events = np.unique(events.numpy())

        if len(unique_events) <= 1:
            # No stratification possible, use random
            return np.random.choice(data.shape[0], max_trials, replace=False)

        # Calculate trials per event type
        trials_per_event = max_trials // len(unique_events)
        remainder = max_trials % len(unique_events)

        selected_indices = []

        for i, event_type in enumerate(unique_events):
            event_indices = np.where(events.numpy() == event_type)[0]

            # Add extra trial to some event types to handle remainder
            n_trials_this_event = trials_per_event + (1 if i < remainder else 0)
            n_trials_this_event = min(n_trials_this_event, len(event_indices))

            if n_trials_this_event > 0:
                selected = np.random.choice(event_indices, n_trials_this_event, replace=False)
                selected_indices.extend(selected)

        return np.array(selected_indices)

    def _quality_based_selection(
            self,
            data: torch.Tensor,
            max_trials: int,
            quality_metric: str = 'variance'
    ) -> np.ndarray:
        """Select trials based on signal quality metrics"""

        if data.shape[0] <= max_trials:
            return np.arange(data.shape[0])

        # Calculate quality metrics for each trial
        if quality_metric == 'variance':
            # Higher variance often indicates more informative trials
            quality_scores = torch.var(data, dim=(1, 2))  # Variance across channels and time
        else:
            # Default to variance
            quality_scores = torch.var(data, dim=(1, 2))

        # Select top trials based on quality
        _, top_indices = torch.topk(quality_scores, max_trials)
        return top_indices.numpy()

    def _load_metadata_cache(self):
        """Load metadata from HDF5 files directly, not JSON"""
        for file_path in self.subject_files:
            try:
                # REPLACE JSON loading with direct HDF5 access
                with h5py.File(file_path, "r") as f:
                    metadata = dict(f.attrs)
                    # CRITICAL: Get channel names from HDF5, not JSON
                    if 'channel_names' in f:
                        metadata['channel_names'] = [ch.decode('utf-8') for ch in f['channel_names'][:]]
                    else:
                        metadata['channel_names'] = []

                self.metadata_cache[str(file_path)] = metadata
            except Exception as e:
                print(f"Error loading metadata for {file_path}: {e}")

    def load_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Memory-efficient batch loading with direct category labels"""
        batch_data = []
        batch_labels = []
        batch_metadata = []
        batch_image_categories = []

        for idx in indices:
            file_path = self.subject_files[idx]

            # Load EEG data from HDF5
            with h5py.File(file_path, "r") as f:
                eeg_data = np.array(f["eeg_data"][:], dtype=np.float32)

            metadata = self.metadata_cache.get(str(file_path), {})
            image_categories = self.image_category_cache.get(str(file_path), pd.DataFrame())

            # Extract labels directly from categories
            if not image_categories.empty:
                n_trials = len(image_categories)

                # Create label dictionary with all relevant category information
                labels = {
                    'category_num': image_categories['category_num'].fillna(0).astype(
                         np.int32).values if 'category_num' in image_categories.columns else np.zeros(n_trials,
                                                                                                     dtype=np.int32),
                    'super_category_id': image_categories['super_category_id'].fillna(0).astype(
                        np.int32).values if 'super_category_id' in image_categories.columns else np.zeros(n_trials,
                                                                                                          dtype=np.int32),
                     'fname': image_categories[
                        'fname'].tolist() if 'fname' in image_categories.columns else [''] * n_trials,
                    # 'image_filename': image_categories[
                    #     'image_filename'].tolist() if 'image_filename' in image_categories.columns else [''] * n_trials,
                }

                # Debug logging for first subject
                if idx == 0:
                    print(f"  DEBUG {file_path.name}: Loaded {n_trials} trials")
                    # print(f"  Sample category_num: {labels['category_num'][:3]}")
                    print(f"  Sample super_category_id: {labels['super_category_id'][:3]}")
                    print(f"  Sample fnames: {labels['image_filename'][:3]}")
            else:
                # Fallback when no category information available
                n_trials = eeg_data.shape[0]
                labels = {
                    'category_num': np.zeros(n_trials, dtype=np.int32),
                    'super_category_id': np.zeros(n_trials, dtype=np.int32),
                    'image_filename': [''] * n_trials,
                }

                if idx == 0:
                    print(f"  WARNING: No image categories for {file_path.name}, using default labels")

            # Convert to tensors
            batch_data.append(torch.from_numpy(eeg_data))
            batch_labels.append(labels)
            batch_metadata.append(metadata)
            batch_image_categories.append(image_categories)

        return {
            "eeg_data": batch_data,
            "labels": batch_labels,  # Changed from "events" to "labels"
            "metadata": batch_metadata,
            "image_categories": batch_image_categories
        }
    # 2. Enhanced prepare_harmonization_batch_enhanced method
    def prepare_harmonization_batch_enhanced(
            self,
            batch: Dict,
            max_trials_per_subject: Optional[int] = None,
            sequence_padding: str = 'max_batch',
            trial_selection: str = 'quality'
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced batch preparation with image category preservation
        """
        all_trials = []
        all_metadata = []
        all_image_categories = []  # NEW: Store selected image categories
        subject_indices = []

        # Collect all trials with their corresponding image categories
        for subj_idx, (data, batch_labels, metadata, img_categories) in enumerate(zip(
                batch["eeg_data"], batch["labels"], batch["metadata"], batch["image_categories"]
        )):
            n_trials, n_channels, n_time = data.shape

            # Apply trial limit if specified
            if max_trials_per_subject is not None:
                max_trials = min(max_trials_per_subject, n_trials)
            else:
                max_trials = n_trials

            if max_trials == 0:
                continue

            if n_trials > max_trials:
                # Simplified trial selection (no event-based stratification)
                if trial_selection == 'quality':
                    selected_trials = self._quality_based_selection(data, max_trials, quality_metric='variance')
                else:  # random
                    selected_trials = np.random.choice(n_trials, max_trials, replace=False)

                selected_data = data[selected_trials]

                # Filter labels for selected trials
                selected_labels = {
                     'category_num': batch_labels['category_num'][selected_trials],
                    'super_category_id': batch_labels['super_category_id'][selected_trials],
                    'fname': [batch_labels['fname'][i] for i in selected_trials],
                    # 'image_filename': [batch_labels['image_filename'][i] for i in selected_trials]
                }

                # Filter image categories by index position
                if not img_categories.empty:
                    selected_img_cats = img_categories.iloc[selected_trials].reset_index(drop=True)
                else:
                    selected_img_cats = pd.DataFrame()

                print(f"    Selected {len(selected_trials)} trials using {trial_selection} strategy")
            else:
                selected_data = data
                selected_trials = np.arange(n_trials)
                selected_labels = batch_labels  # Use batch_labels instead of labels
                selected_img_cats = img_categories.copy() if not img_categories.empty else pd.DataFrame()

            # Add each trial with metadata
            for trial_idx, trial_data in enumerate(selected_data):
                if isinstance(trial_data, torch.Tensor):
                    trial_tensor = trial_data.float()
                else:
                    trial_tensor = torch.from_numpy(trial_data).float()

                all_trials.append(trial_tensor)

                # Enhanced trial metadata
                trial_metadata = metadata.copy()
                trial_metadata["subject_idx"] = subj_idx
                trial_metadata["original_trial_idx"] = int(selected_trials[trial_idx])
                trial_metadata["trial_shape"] = trial_tensor.shape

                # ADD label information to metadata
                trial_metadata["category_num"] = int(selected_labels['category_num'][trial_idx])
                trial_metadata["super_category_id"] = int(selected_labels['super_category_id'][trial_idx])
                trial_metadata["fname"] = selected_labels['fname'][trial_idx]
                # trial_metadata["image_filename"] = selected_labels['image_filename'][trial_idx]


                if not selected_img_cats.empty:
                    # Use direct indexing instead of trial_index matching
                    if trial_idx < len(selected_img_cats):
                        img_info = selected_img_cats.iloc[trial_idx]  # Direct index access
                        trial_metadata.update({

                            "image_category": str(img_info.get('category_name', 'unknown')),
                            "super_category": str(img_info.get('super_category', 'unknown')),
                            "super_category_id": int(img_info.get('super_category_id', 0)) if pd.notna(
                                img_info.get('super_category_id')) else 0,

                            "image_filename": str(img_info.get('fname', '')),
                            # "image_filename": str(img_info.get('image_filename', '')),

                            "image_path": str(img_info.get('image_path_full', '')),
                            # "image_path": str(img_info.get('image_path', '')),

                            # "mapped": bool(img_info.get('mapped', False)),
                            # Addingi new fields from my parquet structure:
                            "minor_category": str(img_info.get('minor_category', 'unknown')),
                            "minor_category_id": int(img_info.get('minor_category_id', 0)) if pd.notna(
                                img_info.get('minor_category_id')) else 0,
                            "category_num": int(img_info.get('category_num', 0)) if pd.notna(
                                img_info.get('category_num')) else 0,
                            "category_img_num": int(img_info.get('category_img_num', 0)) if pd.notna(
                                img_info.get('category_img_num')) else 0,
                            # Handle animate/inanimate for Infant dataset only
                            "sub_category": str(img_info.get('sub_category', '')) if 'sub_category' in img_info else ''
                        })
                    else:
                        # Trial index out of bounds
                        trial_metadata.update({
                            "image_category": 'unknown',
                            "super_category": 'unknown',
                            "super_category_id": 0,

                            "image_filename": '',
                            "image_path": '',
                            # "mapped": False,
                            "minor_category": 'unknown',
                            "minor_category_id": 0,
                            "category_num": 0,
                            "category_img_num": 0,
                            "sub_category": ''
                        })
                else:
                    # No image categories available
                    trial_metadata.update({
                        "image_category": 'unknown',
                        "super_category": 'unknown',
                        "super_category_id": 0,

                        "image_filename": '',
                        "image_path": '',
                        # "mapped": False,
                        "sub_category": ''
                    })
                all_metadata.append(trial_metadata)
                subject_indices.append(subj_idx)

        if len(all_trials) == 0:
            return None

        # Handle sequence padding (existing logic remains the same)
        if sequence_padding == 'max_batch':
            # Pad all sequences to batch maximum
            max_time = max(trial.shape[1] for trial in all_trials)
            max_channels_batch = max(trial.shape[0] for trial in all_trials)

            batch_size = len(all_trials)
            padded_data = torch.zeros(batch_size, self.max_channels, max_time)
            channel_masks = torch.zeros(batch_size, self.max_channels, dtype=torch.bool)
            time_masks = torch.zeros(batch_size, max_time, dtype=torch.bool)
            channel_indices = torch.zeros(batch_size, self.max_channels, dtype=torch.long)

            for i, (trial_data, metadata) in enumerate(zip(all_trials, all_metadata)):
                n_channels, n_time = trial_data.shape

                # Pad trial data
                padded_data[i, :n_channels, :n_time] = trial_data
                channel_masks[i, :n_channels] = True
                time_masks[i, :n_time] = True

                # Set channel indices
                channel_indices[i, :n_channels] = torch.arange(n_channels)

            return {
                "eeg_data": padded_data,
                "channel_mask": channel_masks,
                "time_mask": time_masks,
                "channel_indices": channel_indices,
                "metadata": all_metadata,  # Now includes image categories
                "subject_indices": subject_indices,
                "sequence_lengths": [trial.shape[1] for trial in all_trials],
                "channel_names": [meta.get('channel_names', None) for meta in all_metadata]
            }
        else:
            # Return list format for variable length processing
            return {
                "eeg_data": all_trials,
                "metadata": all_metadata,  # Now includes image categories
                "subject_indices": subject_indices,
                "channel_names": [meta.get('channel_names', None) for meta in all_metadata]
            }

# 4.  function to show image categories
def inspect_harmonized_data(file_path: Path):
    """ function for dual embedding format with image categories"""
    print(f"\nInspecting harmonized data: {file_path.name}")

    try:
        with h5py.File(file_path, 'r') as f:
            # Basic info
            print(f"  Subject ID: {f.attrs.get('subject_id', 'Unknown')}")
            print(f"  Dataset: {f.attrs.get('dataset', 'Unknown')}")
            print(f"  Number of trials: {f.attrs.get('n_trials', 'Unknown')}")

            # NEW: Integrated category info
            print(f"  Unique image categories: {f.attrs.get('unique_image_categories', 'Unknown')}")
            print(f"  Unique super categories: {f.attrs.get('unique_super_categories', 'Unknown')}")
            print(f"  Mapping success rate: {f.attrs.get('mapping_success_rate', 0):.2%}")
            print(f"  Has animate/inanimate: {f.attrs.get('has_sub_category', False)}")

            # Check available datasets
            available_keys = list(f.keys())
            print(f"  Available data: {available_keys}")

            # Inspect embeddings
            if 'embeddings_pooled' in f:
                pooled = f['embeddings_pooled'][:]
                print(f"  Pooled embeddings shape: {pooled.shape}")
                print(f"    Mean: {np.mean(pooled):.4f}, Std: {np.std(pooled):.4f}")

            if 'embeddings_sequence' in f:
                sequence = f['embeddings_sequence'][:]
                print(f"  Sequence embeddings shape: {sequence.shape}")
                print(f"    Mean: {np.mean(sequence):.4f}, Std: {np.std(sequence):.4f}")

            # Show sample integrated categories
            if 'image_categories' in f:
                categories = [cat.decode('utf-8') for cat in f['image_categories'][:5]]
                super_cats = [cat.decode('utf-8') for cat in f['super_categories'][:5]]
                print(f"  Sample image categories: {categories}")
                print(f"  Sample super categories: {super_cats}")

                # Show animate/inanimate if available
                if 'sub_category' in f:
                    animate_info = [cat.decode('utf-8') for cat in f['sub_category'][:5]]
                    print(f"  Sample animate/inanimate: {animate_info}")

    except Exception as e:
        print(f"  Error inspecting file: {e}")

class EnhancedHarmonizationSaver:
    """Enhanced saver with dual embedding format support"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Enhanced save_dual_embeddings method to include image categories
    def save_dual_embeddings(
            self,
            embeddings_pooled: torch.Tensor,
            embeddings_sequence: torch.Tensor,
            metadata_list: List[Dict],
            subject_id: str,
            dataset_name: str
    ):
        """Save both pooled and sequence embeddings with integrated category information"""
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        output_file = dataset_dir / f"{subject_id}_harmonized.h5"

        # Convert tensors to numpy
        def tensor_to_numpy(tensor):
            if isinstance(tensor, torch.Tensor):
                if tensor.is_cuda:
                    return tensor.cpu().detach().numpy()
                else:
                    return tensor.detach().numpy()
            return np.array(tensor)

        pooled_np = tensor_to_numpy(embeddings_pooled)
        sequence_np = tensor_to_numpy(embeddings_sequence)

        # Extract category information from metadata
        all_image_categories = []
        all_super_categories = []
        all_super_category_ids = []
        all_image_filenames = []
        all_image_paths = []
        # all_mapped_flags = []

        #  Only for datasets that have animate/inanimate
        has_sub_category = False   # initialize here

        all_animate_inanimate = []

        for meta in metadata_list:
            # Ensure we're getting actual values, not 'unknown' defaults
            image_cat = meta.get('image_category', 'unknown')  # This now comes from 'category_name'
            super_cat = meta.get('super_category', 'unknown')  # KEEP SAME
            super_cat_id = meta.get('super_category_id', 0)  # KEEP SAME

            # ADD new field extractions:
            minor_cat = meta.get('minor_category', 'unknown')
            minor_cat_id = meta.get('minor_category_id', 0)
            cat_num = meta.get('category_num', 0)
            cat_img_num = meta.get('category_img_num', 0)

            # Append to respective lists:
            all_minor_categories.append(str(minor_cat))
            all_minor_category_ids.append(int(minor_cat_id) if pd.notna(minor_cat_id) else 0)
            # all_category_nums.append(int(cat_num) if pd.notna(cat_num) else 0)
            # all_category_img_nums.append(int(cat_img_num) if pd.notna(cat_img_num) else 0)

            # CHANGE image path access:
            all_image_paths.append(str(meta.get('image_path', '')))  # Now uses 'image_path_full'

            # CHANGE filename access:
            all_image_filenames.append(str(meta.get('image_filename', '')))  # Now uses 'fname'

            # Check for animate/inanimate (Infant dataset specific)
            sub_cat_value = meta.get('sub_category', '')
            if sub_cat_value and sub_cat_value != '' and str(sub_cat_value).lower() not in ['unknown', 'nan']:
                has_sub_category = True
                all_animate_inanimate.append(str(sub_cat_value))
            elif has_sub_category:  # Keep consistent length
                all_animate_inanimate.append('unknown')

        # DEBUG: Final category statistics
        print(f"Final category extraction results:")
        print(f"  Total trials: {len(all_image_categories)}")
        print(f"  Unique image categories: {len(set(all_image_categories))}")
        print(f"  Unique super categories: {len(set(all_super_categories))}")
        print(f"  Sample image categories: {list(set(all_image_categories))[:5]}")
        print(f"  Sample super categories: {list(set(all_super_categories))[:5]}")
        print(f"  'unknown' count in super_categories: {all_super_categories.count('unknown')}")



        print(
            f"Saving integrated embeddings with categories - Pooled: {pooled_np.shape}, Sequence: {sequence_np.shape}")

        try:
            with h5py.File(output_file, 'w') as f:
                # Save both embedding types
                # f.create_dataset('embeddings_pooled', data=pooled_np, compression='gzip')
                # f.create_dataset('embeddings_sequence', data=sequence_np, compression='gzip')
                # OPTIMIZED: Use faster compression and chunking
                compression_opts = {
                    'compression': 'lzf',  # Faster than gzip
                    'shuffle': True,
                    'chunks': True,
                    'fletcher32': False  # Skip checksum for speed
                }

                f.create_dataset('embeddings_pooled', data=pooled_np, **compression_opts)
                f.create_dataset('embeddings_sequence', data=sequence_np, **compression_opts)

                # UNIVERSAL CATEGORY FIELDS (all datasets)
                f.create_dataset('category_name',
                                 data=[cat.encode('utf-8') for cat in all_image_categories])
                f.create_dataset('super_categories',
                                 data=[cat.encode('utf-8') for cat in all_super_categories])
                f.create_dataset('super_category_ids', data=np.array(all_super_category_ids))
                f.create_dataset('image_filenames',
                                 data=[fname.encode('utf-8') for fname in all_image_filenames])
                f.create_dataset('image_paths',
                                 data=[path.encode('utf-8') for path in all_image_paths])
                f.create_dataset('minor_categories',
                                 data=[cat.encode('utf-8') for cat in all_minor_categories])
                f.create_dataset('minor_category_ids', data=np.array(all_minor_category_ids))
                f.create_dataset('category_nums', data=np.array(all_category_nums))
                f.create_dataset('category_img_nums', data=np.array(all_category_img_nums))

                # f.create_dataset('mapped_flags', data=np.array(all_mapped_flags))

                #  Only save if dataset has animate/inanimate
                if has_sub_category:
                    f.create_dataset('animate_inanimate',
                                     data=[cat.encode('utf-8') for cat in all_animate_inanimate])
                    f.attrs['has_sub_category'] = True
                    f.attrs['total_animate'] = sum(1 for x in all_animate_inanimate if x == 'animate')
                    f.attrs['total_inanimate'] = sum(1 for x in all_animate_inanimate if x == 'inanimate')
                else:
                    f.attrs['has_sub_category'] = False

                # Prepare trial metadata (simplified version)
                trial_data = []
                for i, meta in enumerate(metadata_list):
                    trial_info = {
                        # 'trial_idx': i,
                        'subject_idx': meta.get('subject_idx', 0),
                        'original_trial_idx': meta.get('original_trial_idx', i),
                        'n_channels': meta.get('trial_shape', [0, 0])[0] if 'trial_shape' in meta else 0,
                        'n_timepoints': meta.get('trial_shape', [0, 0])[1] if 'trial_shape' in meta else 0,
                    }
                    trial_data.append(trial_info)

                # Convert to structured array for HDF5
                if trial_data:
                    dtype = [
                        # ('trial_idx', 'i4'),
                        ('subject_idx', 'i4'),
                        ('original_trial_idx', 'i4'),
                        ('n_channels', 'i4'),
                        ('n_timepoints', 'i4'),
                    ]

                    structured_array = np.array([
                        (td['subject_idx'], td['original_trial_idx'],
                         td['n_channels'], td['n_timepoints'])
                        for td in trial_data
                    ], dtype=dtype)

                    f.create_dataset('trial_metadata', data=structured_array)

                # Enhanced metadata with category statistics
                f.attrs['subject_id'] = subject_id
                f.attrs['dataset'] = dataset_name
                f.attrs['n_trials'] = len(metadata_list)
                f.attrs['unique_image_categories'] = len(set(all_image_categories))
                f.attrs['unique_super_categories'] = len(set(all_super_categories))
                f.attrs['mapping_success_rate'] = sum(all_mapped_flags) / len(
                    all_mapped_flags) if all_mapped_flags else 0
                #f.attrs['save_timestamp'] = str(pd.Timestamp.now())

        except Exception as e:
            print(f"Error saving harmonized data: {e}")
            raise

        return output_file


def create_enhanced_harmonization_layer(
    max_channels: int = 128,
    virtual_channels: int = 128,
    output_dim: int = 1024,
    max_time_points: int = 512,
    use_spatial_embeddings: bool = True,
    target_montage: str = "standard_1020"  # New parameter
) -> EEGHarmonizationLayer:
    """Factory function to create enhanced harmonization layer with MNE support"""

    config = HarmonizationConfig(
        max_input_channels=max_channels,
        virtual_channels=virtual_channels,
        output_embedding_dim=output_dim,
        max_time_points=max_time_points,
        transformer_layers=6,
        transformer_heads=8,
        transformer_dim=512,
        dropout=0.1,
        use_spatial_embeddings=use_spatial_embeddings,
        target_montage=target_montage  # Pass target montage
    )

    return EEGHarmonizationLayer(config)

def process_datasets_flexible(
        dataset_dirs: List[Path],
        output_dir: Path,
        num_subjects_per_dataset: Optional[int] = None,
        max_trials_per_subject: Optional[int] = None,
        embedding_dim: int = 1024,
        sequence_padding: str = 'max_batch',
        use_spatial_embeddings: bool = True,
        target_montage: str = "standard_1020",
        trial_selection: str = 'quality'  # ADD THIS LINE
):
    """
    Flexible processing function with configurable parameters

    Args:
        dataset_dirs: List of dataset directories to process
        output_dir: Output directory for harmonized data
        num_subjects_per_dataset: Max subjects per dataset (None = all)
        max_trials_per_subject: Max trials per subject (None = all)
        embedding_dim: Fixed embedding dimension (default 1024)
        sequence_padding: 'max_batch' or 'individual'
        use_spatial_embeddings: Whether to use spatial electrode positions
    """
    #print("Starting flexible harmonization processing...")
    print(f"Datasets to process: {len(dataset_dirs)}")
    print(f"Subjects per dataset: {num_subjects_per_dataset or 'All'}")
    print(f"Trials per subject: {max_trials_per_subject or 'All'}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Spatial embeddings: {use_spatial_embeddings}")

    # Create enhanced harmonizer
    # Create enhanced harmonizer
    harmonizer = create_enhanced_harmonization_layer(
        max_channels=128,
        virtual_channels=128,
        output_dim=embedding_dim,
        max_time_points=512,
        use_spatial_embeddings=use_spatial_embeddings
    )

    # OPTIMIZED: Setup device with mixed precision
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None

    # OPTIMIZATION: Model compilation (PyTorch 2.0+)
    # try:
    #     harmonizer = torch.compile(harmonizer, mode='max-autotune')
    #     print("Model compiled with PyTorch 2.0 optimization")
    # except:
    #     print("PyTorch compilation not available, using standard mode")
    print("Skipping torch.compile in parallel mode for MNE compatibility")
    # OPTIMIZATION: Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        harmonizer = torch.nn.DataParallel(harmonizer)
        device = torch.device('cuda:0')

    harmonizer = harmonizer.to(device)
    harmonizer.eval()  # Set to eval mode for inference
    print(f"Model loaded on {device} with mixed precision: {use_amp}")

    # Create saver
    saver = EnhancedHarmonizationSaver(output_dir)
    all_processed_files = []

    # Process each dataset
    for dataset_idx, data_dir in enumerate(dataset_dirs):
        if not data_dir.exists():
            print(f"Directory {data_dir} not found, skipping")
            continue

        dataset_name = data_dir.name
        loader = EEGDataLoader(data_dir, batch_size=1, max_channels=128)

        total_subjects = len(loader.subject_files)
        if num_subjects_per_dataset is not None:
            total_subjects = min(total_subjects, num_subjects_per_dataset)

        print(f"\nProcessing {dataset_name} dataset ({total_subjects} subjects)")
        dataset_processed_files = []

        # Process each subject with progress tracking
        with tqdm(range(total_subjects), desc=f"Dataset {dataset_idx + 1}/{len(dataset_dirs)}: {dataset_name}") as pbar:
            for subject_idx in pbar:
                try:
                    start_time = time.time()

                    # Load subject data
                    batch = loader.load_batch([subject_idx])
                    harmonized_batch = loader.prepare_harmonization_batch_enhanced(
                        batch,
                        max_trials_per_subject=max_trials_per_subject,
                        sequence_padding=sequence_padding,
                        trial_selection=trial_selection
                    )

                    if harmonized_batch is None:
                        print(f"No valid trials for subject {subject_idx}, skipping")
                        continue

                    # Get subject info
                    subject_file = loader.subject_files[subject_idx]
                    subject_id = subject_file.stem.replace('_processed', '')

                    n_trials = harmonized_batch['eeg_data'].shape[0]
                    pbar.set_postfix({
                        'subject': subject_id[:10],
                        'trials': n_trials
                    })

                    # Move to GPU
                    gpu_batch = {}
                    for key, value in harmonized_batch.items():
                        if isinstance(value, torch.Tensor):
                            gpu_batch[key] = value.to(device)
                        else:
                            gpu_batch[key] = value

                    # Process in chunks to manage memory
                    # chunk_size = 32
                    # OPTIMIZED: Dynamic chunk size based on available GPU memory
                    if torch.cuda.is_available():
                        available_memory = torch.cuda.get_device_properties(
                            0).total_memory - torch.cuda.memory_allocated()
                        estimated_memory_per_trial = gpu_batch['eeg_data'].element_size() * gpu_batch['eeg_data'][
                            0].numel() * 4
                        optimal_chunk_size = min(128, max(16, int(available_memory * 0.7 / estimated_memory_per_trial)))
                    else:
                        optimal_chunk_size = 32

                    print(f"  Using dynamic chunk size: {optimal_chunk_size}")
                    batch_size = gpu_batch['eeg_data'].shape[0]
                    all_embeddings_pooled = []
                    all_embeddings_sequence = []

                    for i in range(0, batch_size, optimal_chunk_size):
                        end_idx = min(i + optimal_chunk_size, batch_size)

                        chunk_data = gpu_batch['eeg_data'][i:end_idx]
                        chunk_channel_mask = gpu_batch['channel_mask'][i:end_idx]
                        chunk_time_mask = gpu_batch['time_mask'][i:end_idx]
                        chunk_channel_indices = gpu_batch['channel_indices'][i:end_idx]

                        # Get channel names for chunk
                        chunk_channels = None
                        if harmonized_batch['channel_names']:
                            for cn in harmonized_batch['channel_names'][i:end_idx]:
                                if cn is not None:
                                    chunk_channels = cn
                                    break

                        # OPTIMIZED: Mixed precision inference
                        with torch.no_grad():
                            if use_amp:
                                # with autocast():
                                with torch.amp.autocast('cuda'):
                                    output = harmonizer(
                                        chunk_data,
                                        channel_mask=chunk_channel_mask,
                                        time_mask=chunk_time_mask,
                                        channel_indices=chunk_channel_indices,
                                        channel_names=chunk_channels,
                                        output_type='both',
                                        return_attention=False
                                    )
                            else:
                                output = harmonizer(
                                    chunk_data,
                                    channel_mask=chunk_channel_mask,
                                    time_mask=chunk_time_mask,
                                    channel_indices=chunk_channel_indices,
                                    channel_names=chunk_channels,
                                    output_type='both',
                                    return_attention=False
                                )

                        # Store embeddings
                        all_embeddings_pooled.append(output['embedding_pooled'].cpu())
                        all_embeddings_sequence.append(output['embedding_sequence'].cpu())

                        # OPTIMIZED: Explicit memory cleanup
                        del chunk_data, chunk_channel_mask, chunk_time_mask, chunk_channel_indices, output

                    # Combine all chunks
                    final_embeddings_pooled = torch.cat(all_embeddings_pooled, dim=0)
                    final_embeddings_sequence = torch.cat(all_embeddings_sequence, dim=0)

                    # Save dual embeddings
                    output_file = saver.save_dual_embeddings(
                        final_embeddings_pooled,
                        final_embeddings_sequence,
                        harmonized_batch['metadata'],
                        subject_id,
                        dataset_name
                    )

                    if output_file is not None:
                        dataset_processed_files.append(output_file)

                    processing_time = time.time() - start_time

                    # Clear GPU memory
                    # del gpu_batch, all_embeddings_pooled, all_embeddings_sequence
                    # del final_embeddings_pooled, final_embeddings_sequence
                    # torch.cuda.empty_cache()
                    # OPTIMIZED: More aggressive memory cleanup

                    processing_time = time.time() - start_time
                    del gpu_batch, harmonized_batch
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    # Log memory usage
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.max_memory_allocated() / 1e9
                        print(f"  Max GPU memory used: {memory_used:.2f}GB")
                        torch.cuda.reset_peak_memory_stats()

                except Exception as e:
                    print(f"Error processing subject {subject_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        print(f"Dataset {dataset_name} complete: {len(dataset_processed_files)} subjects processed")
        all_processed_files.extend(dataset_processed_files)

    # Final summary
    print(f"\nProcessing complete!")
    print(f"Total files processed: {len(all_processed_files)}")
    print(f"Output directory: {output_dir}")

    return all_processed_files


def process_single_subject_task(args_tuple):
    """Standalone function for parallel processing (must be at module level for pickling)"""
    #"""Standalone function for parallel processing with proper argument handling"""

    # Unpack arguments
    dataset_dir, subject_idx, dataset_name, output_dir, processing_kwargs = args_tuple

    # CRITICAL: Set CUDA environment variables BEFORE importing torch modules
    os.environ['CUDA_VISIBLE_DEVICES'] = str(
        subject_idx % torch.cuda.device_count()) if torch.cuda.is_available() else ''
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    # Disable NVML to avoid the error you're seeing
    os.environ['CUDA_NVML_DISABLED'] = '1'

    # Import torch modules AFTER setting environment
    # import torch._dynamo
    # torch._dynamo.config.suppress_errors = True

    try:
        print(f"Worker processing {dataset_name} subject {subject_idx}")

        # Initialize components in worker process
        from harmonise_new import EEGDataLoader, EnhancedHarmonizationSaver, create_enhanced_harmonization_layer

        loader = EEGDataLoader(Path(dataset_dir), batch_size=1)
        saver = EnhancedHarmonizationSaver(Path(output_dir))

        # Verify subject index is valid
        if subject_idx >= len(loader.subject_files):
            print(f"Subject index {subject_idx} out of range for {len(loader.subject_files)} files")
            return None

        # Create harmonizer in worker process
        harmonizer = create_enhanced_harmonization_layer(
            max_channels=128,
            virtual_channels=128,
            output_dim=processing_kwargs.get('embedding_dim', 1024),
            max_time_points=512,
            use_spatial_embeddings=processing_kwargs.get('use_spatial_embeddings', True)
        )

        # Setup device - Use CPU for parallel to avoid GPU conflicts
        device = torch.device('cpu')  # FORCE CPU for parallel processing
        harmonizer = harmonizer.to(device)
        harmonizer.eval()

        print(f"Processing subject {subject_idx} on {device}")

        # Load ONLY the specified subject
        batch = loader.load_batch([subject_idx])
        if not batch or not batch['eeg_data']:
            print(f"No data loaded for subject {subject_idx}")
            return None

        harmonized_batch = loader.prepare_harmonization_batch_enhanced(
            batch,
            max_trials_per_subject=processing_kwargs.get('max_trials_per_subject'),
            sequence_padding=processing_kwargs.get('sequence_padding', 'max_batch'),
            trial_selection=processing_kwargs.get('trial_selection', 'quality')
        )

        if harmonized_batch is None:
            print(f"No harmonized data for subject {subject_idx}")
            return None

        # Process on CPU (no GPU transfers needed)
        cpu_batch = {}
        for key, value in harmonized_batch.items():
            if isinstance(value, torch.Tensor):
                cpu_batch[key] = value.to(device)
            else:
                cpu_batch[key] = value

        # Process in smaller chunks for CPU
        chunk_size = 16  # Smaller chunks for CPU processing
        batch_size = cpu_batch['eeg_data'].shape[0]
        all_embeddings_pooled = []
        all_embeddings_sequence = []

        print(f"Processing {batch_size} trials in chunks of {chunk_size}")

        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)

            chunk_data = cpu_batch['eeg_data'][i:end_idx]
            chunk_channel_mask = cpu_batch['channel_mask'][i:end_idx]
            chunk_time_mask = cpu_batch['time_mask'][i:end_idx]
            chunk_channel_indices = cpu_batch['channel_indices'][i:end_idx]

            # Get channel names
            chunk_channels = None
            if harmonized_batch['channel_names']:
                for cn in harmonized_batch['channel_names'][i:end_idx]:
                    if cn is not None:
                        chunk_channels = cn
                        break

            # Process with CPU (no mixed precision)
            with torch.no_grad():
                output = harmonizer(
                    chunk_data,
                    channel_mask=chunk_channel_mask,
                    time_mask=chunk_time_mask,
                    channel_indices=chunk_channel_indices,
                    channel_names=chunk_channels,
                    output_type='both',
                    return_attention=False
                )

            all_embeddings_pooled.append(output['embedding_pooled'])
            all_embeddings_sequence.append(output['embedding_sequence'])

            # Memory cleanup
            del chunk_data, chunk_channel_mask, chunk_time_mask, chunk_channel_indices, output

        # Combine results
        final_embeddings_pooled = torch.cat(all_embeddings_pooled, dim=0)
        final_embeddings_sequence = torch.cat(all_embeddings_sequence, dim=0)

        # Get subject ID
        subject_file = loader.subject_files[subject_idx]
        subject_id = subject_file.stem.replace('_processed', '')

        print(
            f"Saving results for {subject_id}: pooled={final_embeddings_pooled.shape}, sequence={final_embeddings_sequence.shape}")

        # Save results
        output_file = saver.save_dual_embeddings(
            final_embeddings_pooled,
            final_embeddings_sequence,
            harmonized_batch['metadata'],
            subject_id,
            dataset_name
        )

        # Final cleanup
        del cpu_batch, all_embeddings_pooled, all_embeddings_sequence
        del final_embeddings_pooled, final_embeddings_sequence

        print(f"Successfully processed {dataset_name} subject {subject_idx} -> {subject_id}")
        return output_file

    except Exception as e:
        print(f"Error processing {dataset_name} subject {subject_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_datasets_parallel(
        dataset_dirs: List[Path],
        output_dir: Path,
        num_workers: int = 4,
        **kwargs
):
    """ Process subjects in parallel with proper task distribution"""

    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # Collect all subject tasks - FIXED: No duplication
    all_tasks = []
    task_info = []  # For debugging

    for dataset_idx, dataset_dir in enumerate(dataset_dirs):
        if not dataset_dir.exists():
            print(f"Directory {dataset_dir} not found, skipping")
            continue

        # Create a temporary loader just to count subjects
        from harmonise_new import EEGDataLoader
        temp_loader = EEGDataLoader(dataset_dir, batch_size=1)
        dataset_name = dataset_dir.name

        total_subjects = len(temp_loader.subject_files)
        if kwargs.get('num_subjects_per_dataset'):
            total_subjects = min(total_subjects, kwargs['num_subjects_per_dataset'])

        print(f"Dataset {dataset_name}: {total_subjects} subjects to process")

        # Create one task per subject for Proper indexing
        for subject_idx in range(total_subjects):
            task = (str(dataset_dir), subject_idx, dataset_name, str(output_dir), kwargs)
            all_tasks.append(task)
            task_info.append(f"{dataset_name}/subject_{subject_idx}")

        del temp_loader  # Cleanup

    print(f"Created {len(all_tasks)} tasks across {len(dataset_dirs)} datasets")
    print(f"Sample tasks: {task_info[:5]}")
    print(f"Processing with {num_workers} workers")

    # Process in parallel using the fixed function
    processed_files = []
    failed_tasks = []

    with ProcessPoolExecutor(max_workers=num_workers,
                             mp_context=mp.get_context('spawn')) as executor:

        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_subject_task, task): task_info[i]
            for i, task in enumerate(all_tasks)
        }

        # Process completed tasks
        for future in tqdm(as_completed(future_to_task),
                           total=len(all_tasks),
                           desc="Processing subjects"):
            task_name = future_to_task[future]
            try:
                result = future.result(timeout=300)  # 5 minute timeout per subject
                if result:
                    processed_files.append(result)
                    print(f"✓ Completed: {task_name}")
                else:
                    failed_tasks.append(task_name)
                    print(f"✗ Failed: {task_name}")
            except Exception as e:
                failed_tasks.append(task_name)
                print(f"✗ Worker error for {task_name}: {e}")

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(processed_files)} subjects")
    print(f"Failed: {len(failed_tasks)} subjects")

    if failed_tasks:
        print(f"Failed tasks: {failed_tasks[:5]}{'...' if len(failed_tasks) > 5 else ''}")

    return processed_files


# Alternative GPU-based parallel processing (if you want to use GPU)
def process_datasets_parallel_gpu(
        dataset_dirs: List[Path],
        output_dir: Path,
        num_workers: int = None,
        **kwargs
):
    """GPU-based parallel processing - one worker per GPU"""

    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU parallel processing")
        return process_datasets_parallel(dataset_dirs, output_dir, 4, **kwargs)

    num_gpus = torch.cuda.device_count()
    if num_workers is None:
        num_workers = num_gpus

    print(f"Using GPU parallel processing with {num_workers} workers on {num_gpus} GPUs")

    # Collect all tasks
    all_tasks = []
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue

        from harmonise_new import EEGDataLoader
        temp_loader = EEGDataLoader(dataset_dir, batch_size=1)
        dataset_name = dataset_dir.name
        total_subjects = len(temp_loader.subject_files)

        if kwargs.get('num_subjects_per_dataset'):
            total_subjects = min(total_subjects, kwargs['num_subjects_per_dataset'])

        for subject_idx in range(total_subjects):
            gpu_id = len(all_tasks) % num_gpus  # Distribute across GPUs
            task = (str(dataset_dir), subject_idx, dataset_name, str(output_dir), kwargs, gpu_id)
            all_tasks.append(task)

        del temp_loader

    # Process with GPU assignment
    processed_files = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_subject_gpu_task, task) for task in all_tasks]

        for future in tqdm(as_completed(futures), total=len(all_tasks), desc="GPU Processing"):
            try:
                result = future.result()
                if result:
                    processed_files.append(result)
            except Exception as e:
                print(f"GPU worker failed: {e}")

    return processed_files


def process_single_subject_gpu_task(args_tuple):
    """GPU version with proper GPU assignment"""
    dataset_dir, subject_idx, dataset_name, output_dir, kwargs, gpu_id = args_tuple

    # Set specific GPU BEFORE importing torch modules
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['CUDA_NVML_DISABLED'] = '1'  # Disable NVML
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'

    # Import torch modules AFTER setting environment
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    try:
        print(f"GPU Worker {gpu_id} processing {dataset_name} subject {subject_idx}")

        # Initialize components in worker process
        from harmonise_new import EEGDataLoader, EnhancedHarmonizationSaver, create_enhanced_harmonization_layer

        loader = EEGDataLoader(Path(dataset_dir), batch_size=1)
        saver = EnhancedHarmonizationSaver(Path(output_dir))

        # Verify subject index is valid
        if subject_idx >= len(loader.subject_files):
            print(f"Subject index {subject_idx} out of range for {len(loader.subject_files)} files")
            return None

        # Create harmonizer in worker process
        harmonizer = create_enhanced_harmonization_layer(
            max_channels=128,
            virtual_channels=128,
            output_dim=kwargs.get('embedding_dim', 1024),
            max_time_points=512,
            use_spatial_embeddings=kwargs.get('use_spatial_embeddings', True)
        )

        # Setup GPU device
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        use_amp = device.type == 'cuda'

        # Move model to assigned GPU
        harmonizer = harmonizer.to(device)
        harmonizer.eval()

        print(f"GPU {gpu_id}: Processing subject {subject_idx} on {device}")

        # Load ONLY the specified subject
        batch = loader.load_batch([subject_idx])
        if not batch or not batch['eeg_data']:
            print(f"No data loaded for subject {subject_idx}")
            return None

        harmonized_batch = loader.prepare_harmonization_batch_enhanced(
            batch,
            max_trials_per_subject=kwargs.get('max_trials_per_subject'),
            sequence_padding=kwargs.get('sequence_padding', 'max_batch'),
            trial_selection=kwargs.get('trial_selection', 'quality')
        )

        if harmonized_batch is None:
            print(f"No harmonized data for subject {subject_idx}")
            return None

        # Move batch to GPU
        gpu_batch = {}
        for key, value in harmonized_batch.items():
            if isinstance(value, torch.Tensor):
                gpu_batch[key] = value.to(device)
            else:
                gpu_batch[key] = value

        # Calculate optimal chunk size for this specific GPU
        if torch.cuda.is_available():
            # Get GPU memory info
            torch.cuda.empty_cache()
            available_memory = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(
                gpu_id)

            # Estimate memory per trial (conservative)
            sample_trial = gpu_batch['eeg_data'][0]
            estimated_memory_per_trial = sample_trial.element_size() * sample_trial.numel() * 8  # 8x safety factor

            # Calculate optimal chunk size (use 60% of available memory)
            optimal_chunk_size = min(64, max(8, int(available_memory * 0.6 / estimated_memory_per_trial)))

            print(
                f"GPU {gpu_id}: Using chunk size {optimal_chunk_size} (available memory: {available_memory / 1e9:.1f}GB)")
        else:
            optimal_chunk_size = 16

        # Process in chunks
        batch_size = gpu_batch['eeg_data'].shape[0]
        all_embeddings_pooled = []
        all_embeddings_sequence = []

        print(f"GPU {gpu_id}: Processing {batch_size} trials in chunks of {optimal_chunk_size}")

        for i in range(0, batch_size, optimal_chunk_size):
            end_idx = min(i + optimal_chunk_size, batch_size)

            chunk_data = gpu_batch['eeg_data'][i:end_idx]
            chunk_channel_mask = gpu_batch['channel_mask'][i:end_idx]
            chunk_time_mask = gpu_batch['time_mask'][i:end_idx]
            chunk_channel_indices = gpu_batch['channel_indices'][i:end_idx]

            # Get channel names
            chunk_channels = None
            if harmonized_batch['channel_names']:
                for cn in harmonized_batch['channel_names'][i:end_idx]:
                    if cn is not None:
                        chunk_channels = cn
                        break

            # Process with mixed precision if using CUDA
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        output = harmonizer(
                            chunk_data,
                            channel_mask=chunk_channel_mask,
                            time_mask=chunk_time_mask,
                            channel_indices=chunk_channel_indices,
                            channel_names=chunk_channels,
                            output_type='both',
                            return_attention=False
                        )
                else:
                    output = harmonizer(
                        chunk_data,
                        channel_mask=chunk_channel_mask,
                        time_mask=chunk_time_mask,
                        channel_indices=chunk_channel_indices,
                        channel_names=chunk_channels,
                        output_type='both',
                        return_attention=False
                    )

            # Move results to CPU immediately to free GPU memory
            all_embeddings_pooled.append(output['embedding_pooled'].cpu())
            all_embeddings_sequence.append(output['embedding_sequence'].cpu())

            # Aggressive GPU memory cleanup
            del chunk_data, chunk_channel_mask, chunk_time_mask, chunk_channel_indices, output
            torch.cuda.empty_cache()

        # Combine results (on CPU)
        final_embeddings_pooled = torch.cat(all_embeddings_pooled, dim=0)
        final_embeddings_sequence = torch.cat(all_embeddings_sequence, dim=0)

        # Get subject ID
        subject_file = loader.subject_files[subject_idx]
        subject_id = subject_file.stem.replace('_processed', '')

        print(
            f"GPU {gpu_id}: Saving results for {subject_id}: pooled={final_embeddings_pooled.shape}, sequence={final_embeddings_sequence.shape}")

        # Save results
        output_file = saver.save_dual_embeddings(
            final_embeddings_pooled,
            final_embeddings_sequence,
            harmonized_batch['metadata'],
            subject_id,
            dataset_name
        )

        # Final cleanup
        del gpu_batch, all_embeddings_pooled, all_embeddings_sequence
        del final_embeddings_pooled, final_embeddings_sequence
        torch.cuda.empty_cache()

        # Log GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated(gpu_id) / 1e9
            print(f"GPU {gpu_id}: Max memory used: {memory_used:.2f}GB")
            torch.cuda.reset_peak_memory_stats(gpu_id)

        print(f"GPU {gpu_id}: Successfully processed {dataset_name} subject {subject_idx} -> {subject_id}")
        return output_file

    except torch.cuda.OutOfMemoryError as e:
        print(f"GPU {gpu_id}: Out of memory for subject {subject_idx}. Try reducing chunk size or using CPU.")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"GPU {gpu_id}: Error processing {dataset_name} subject {subject_idx}: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def main():
    """Main function with argument parsing"""
    import random
    torch.manual_seed(44)
    np.random.seed(44)
    random.seed(44)

    parser = argparse.ArgumentParser(description="Enhanced EEG Harmonization")
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Paths to dataset directories')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for harmonized data')
    parser.add_argument('--num_subjects', type=int, default=None,
                        help='Number of subjects per dataset (default: all)')
    parser.add_argument('--max_trials', type=int, default=None,
                        help='Max trials per subject (default: all)')
    parser.add_argument('--embedding_dim', type=int, default=1024,
                        help='Embedding dimension (default: 1024)')
    parser.add_argument('--trial_selection', choices=['random', 'stratified', 'quality'],
                        default='quality', help='Trial selection strategy when limiting trials')
    parser.add_argument('--sequence_padding', choices=['max_batch', 'individual'],
                        default='max_batch', help='Sequence padding strategy')
    parser.add_argument('--target_montage', type=str, default='standard_1020',
                        help='Target montage for harmonization (default: standard_1020)')
    parser.add_argument('--debug_channels', action='store_true',
                        help='Debug channel mapping and exit')
    parser.add_argument('--use_spatial', action='store_true', default=True,
                        help='Use spatial electrode embeddings')
    parser.add_argument('--inspect_only', action='store_true',
                        help='Only inspect existing files')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='Use parallel processing across multiple workers')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--gpu_parallel', action='store_true', default=False,
                        help='Use GPU parallel processing (one worker per GPU)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.inspect_only:
        # Inspection mode
        saved_files = list(output_dir.rglob("*.h5"))
        if saved_files:
            print(f"Found {len(saved_files)} saved files")
            for file_path in saved_files[:5]:  # Inspect first 5
                inspect_harmonized_data(file_path)
        else:
            print("No saved files found.")
        return

        # Convert string paths to Path objects
    dataset_dirs = [Path(d) for d in args.datasets]

    # Choose processing method
    if args.gpu_parallel:
        print("Using GPU parallel processing...")
        processed_files = process_datasets_parallel_gpu(
            dataset_dirs=dataset_dirs,
            output_dir=output_dir,
            num_workers=args.num_workers,
            num_subjects_per_dataset=args.num_subjects,
            max_trials_per_subject=args.max_trials,
            embedding_dim=args.embedding_dim,
            sequence_padding=args.sequence_padding,
            use_spatial_embeddings=args.use_spatial,
            trial_selection=args.trial_selection
        )
    elif args.parallel:
        print("Using CPU parallel processing...")
        processed_files = process_datasets_parallel(
            dataset_dirs=dataset_dirs,
            output_dir=output_dir,
            num_workers=args.num_workers,
            num_subjects_per_dataset=args.num_subjects,
            max_trials_per_subject=args.max_trials,
            embedding_dim=args.embedding_dim,
            sequence_padding=args.sequence_padding,
            use_spatial_embeddings=args.use_spatial,
            trial_selection=args.trial_selection
        )
    else:
        print("Using sequential processing...")
        from harmonise_new import process_datasets_flexible
        processed_files = process_datasets_flexible(
            dataset_dirs=dataset_dirs,
            output_dir=output_dir,
            num_subjects_per_dataset=args.num_subjects,
            max_trials_per_subject=args.max_trials,
            embedding_dim=args.embedding_dim,
            sequence_padding=args.sequence_padding,
            use_spatial_embeddings=args.use_spatial,
            trial_selection=args.trial_selection
        )

    # Inspect some results
    if processed_files:
        print("\nSample results:")
        for file_path in processed_files[:3]:
            from harmonise_new import inspect_harmonized_data
            inspect_harmonized_data(file_path)

if __name__ == "__main__":
    main()




def load_harmonized_data_example(file_path: Path) -> Dict[str, np.ndarray]:
    """Example function showing how to load the harmonized data"""

    with h5py.File(file_path, 'r') as f:
        data = {
            'embeddings_pooled': f['embeddings_pooled'][:],  # (n_trials, 1024)
            'embeddings_sequence': f['embeddings_sequence'][:],  # (n_trials, seq_len, 1024)
        }

        # Load metadata
        if 'trial_metadata' in f:
            data['trial_metadata'] = f['trial_metadata'][:]

        if 'channel_names' in f:
            data['channel_names'] = [name.decode('utf-8') for name in f['channel_names'][:]]

        # Load attributes
        data['subject_metadata'] = dict(f.attrs)

    return data


# Alternative API for programmatic use
class HarmonizationPipeline:
    """High-level api for EEG harmonization"""

    def __init__(
            self,
            embedding_dim: int = 1024,
            use_spatial_embeddings: bool = True,
            device: str = 'auto'
    ):
        self.embedding_dim = embedding_dim
        self.use_spatial_embeddings = use_spatial_embeddings

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.harmonizer = self._create_harmonizer()

    def _create_harmonizer(self):
        harmonizer = create_enhanced_harmonization_layer(
            max_channels=128,
            virtual_channels=128,
            output_dim=self.embedding_dim,
            use_spatial_embeddings=self.use_spatial_embeddings
        )

        if torch.cuda.device_count() > 1:
            harmonizer = torch.nn.DataParallel(harmonizer)

        return harmonizer.to(self.device)

    def process_datasets(
            self,
            dataset_paths: List[str],
            output_dir: str,
            num_subjects_per_dataset: Optional[int] = None,
            max_trials_per_subject: Optional[int] = None
    ) -> List[Path]:
        """Process multiple datasets"""

        dataset_dirs = [Path(p) for p in dataset_paths]
        output_path = Path(output_dir)

        return process_datasets_flexible(
            dataset_dirs=dataset_dirs,
            output_dir=output_path,
            num_subjects_per_dataset=num_subjects_per_dataset,
            max_trials_per_subject=max_trials_per_subject,
            embedding_dim=self.embedding_dim,
            use_spatial_embeddings=self.use_spatial_embeddings
        )

    def harmonize_single_trial(
            self,
            eeg_data: np.ndarray,
            channel_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Harmonize a single trial for real-time use"""

        # Convert to tensor and add batch dimension
        eeg_tensor = torch.from_numpy(eeg_data).float().unsqueeze(0)
        eeg_tensor = eeg_tensor.to(self.device)

        # Create masks
        n_channels, n_time = eeg_data.shape
        channel_mask = torch.ones(1, n_channels, dtype=torch.bool, device=self.device)
        time_mask = torch.ones(1, n_time, dtype=torch.bool, device=self.device)
        channel_indices = torch.arange(n_channels, device=self.device).unsqueeze(0)

        with torch.no_grad():
            output = self.harmonizer(
                eeg_tensor,
                channel_mask=channel_mask,
                time_mask=time_mask,
                channel_indices=channel_indices,
                channel_names=channel_names,
                output_type='both'
            )

        return {
            'embedding_pooled': output['embedding_pooled'].cpu().numpy(),
            'embedding_sequence': output['embedding_sequence'].cpu().numpy()
        }


def plot_embedding_umap(harmonizer, data_loader, num_subjects=5):
    """Plot UMAP embeddings colored by subject vs category"""
    import umap
    import matplotlib.pyplot as plt

    harmonizer.eval()
    embeddings, subjects, categories, montages = [], [], [], []

    with torch.no_grad():
        for i in range(min(num_subjects, len(data_loader.subject_files))):
            batch = data_loader.load_batch([i])
            harmonized_batch = data_loader.prepare_harmonization_batch_enhanced(batch)

            if harmonized_batch is None:
                continue

            # Get embeddings
            output = harmonizer(
                harmonized_batch['eeg_data'],
                channel_mask=harmonized_batch['channel_mask'],
                channel_names=harmonized_batch['channel_names'][0],
                output_type='pooled'
            )

            emb = output['embedding_pooled'].cpu().numpy()
            embeddings.append(emb)

            # Get metadata
            for meta in harmonized_batch['metadata']:
                subjects.append(meta.get('subject_idx', i))
                categories.append(meta.get('super_category', 'unknown'))
                montages.append('unknown')  # Can extract from channel names if needed

    if not embeddings:
        print("No embeddings to plot")
        return

    embeddings = np.vstack(embeddings)
    umap_emb = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot by subject
    scatter = axes[0].scatter(umap_emb[:, 0], umap_emb[:, 1], c=subjects, alpha=0.6)
    axes[0].set_title("UMAP by Subject")
    plt.colorbar(scatter, ax=axes[0])

    # Plot by category
    unique_cats = list(set(categories))
    cat_colors = [unique_cats.index(c) for c in categories]
    scatter = axes[1].scatter(umap_emb[:, 0], umap_emb[:, 1], c=cat_colors, alpha=0.6)
    axes[1].set_title("UMAP by Category")
    plt.colorbar(scatter, ax=axes[1])

    # Plot by montage (placeholder)
    axes[2].scatter(umap_emb[:, 0], umap_emb[:, 1], alpha=0.6)
    axes[2].set_title("UMAP by Montage")

    plt.tight_layout()
    plt.savefig("/raid/datasets/tanaya/fm/phase2_f/harmonized_embeddings.png")


def plot_anatomical_attention_maps(harmonizer, sample_eeg, channel_names, channel_positions):
    """Plot scalp maps showing anatomical attention patterns"""
    attention_weights = harmonizer.lcm._map_to_anatomical_space(
        sample_eeg, None, None, None, channel_positions
    )[1]  # Get anatomical attention

    # Plot first 9 anatomical queries
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(9):
        ax = axes[i // 3, i % 3]
        weights = attention_weights[0, i, :].cpu().numpy()

        # Create simple scalp plot (you'll need MNE for proper topomap)
        scatter = ax.scatter(
            channel_positions[0, :, 0], channel_positions[0, :, 1],
            c=weights, s=50, alpha=0.8
        )
        ax.set_title(f'Anatomical Query {i}')
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.savefig("/raid/datasets/tanaya/fm/phase2_f/anatomical_attention_maps.png")
