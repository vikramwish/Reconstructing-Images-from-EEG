# Reconstructing Images from EEG

This repository implements multiple approaches for **reconstructing visual images from EEG (electroencephalography) brain signals**. It explores how neural activity recorded during visual perception can be decoded and used to generate the images a person was viewing.

The project contains three complementary sub-projects, each tackling the problem from a different angle:

1. **GED** — Classical signal decomposition to extract individualized EEG frequencies
2. **EEG-Image Reconstruction Foundation Model** — A 4-stage deep learning pipeline trained on 1.6M EEG trials
3. **Reconstruction with Bands** — Frequency band and electrode analysis for targeted EEG decoding

---

## Repository Structure

```
Reconstructing-Images-from-EEG/
├── README.md                                       # This file
├── LICENSE                                         # GPLv3
│
├── GED/                                            # EEG frequency analysis (MATLAB + Python)
│   ├── Readme.md
│   ├── npyTomat.py                                 # Convert .npy EEG data to .mat format
│   ├── ged_analysis.m                              # GED analysis (MATLAB)
│   ├── filterFGx.m                                 # Frequency filter utility (MATLAB)
│   └── topoplot_eeg63.m                            # Topographic plotting (MATLAB)
│
├── EEG-Image Reconstruction Foundation model/      # 4-stage foundation model pipeline
│   ├── Readme.md
│   ├── environment.yaml                            # Conda environment (Python 3.9, CUDA 11.8)
│   ├── preprocessing.py                            # Raw EEG data loading and preprocessing
│   ├── preprocess_all.py                           # Preprocessing orchestrator
│   ├── stage1_harmonise.py                         # Stage 1: Multi-dataset EEG harmonization
│   ├── stage2_eeg_vit.py                           # Stage 2: EEG-ViT encoder with MoE
│   ├── stage3_alignment.py                         # Stage 3: EEG–Image contrastive alignment
│   ├── extract_stage3.py                           # Extract aligned embeddings
│   └── stage4_reconstruct.py                       # Stage 4: Diffusion-based image reconstruction
│
└── Reconstruction with Bands/                      # Frequency band analysis approach
    ├── README.md
    ├── preparation/                                # Data preparation and feature extraction
    │   ├── requirements.txt                        # Python dependencies
    │   ├── prepare_thingseeg2_data.py              # ThingsEEG2 dataset preparation
    │   ├── save_thingseeg2_images.py               # Save training/test images
    │   ├── save_thingseeg2_concepts.py             # Extract image concepts
    │   ├── process.py                              # Core EEG epoching with MNE
    │   ├── process_bands.py                        # Frequency band filtering
    │   ├── process_electrodes.py                   # Electrode-specific processing
    │   ├── run_bands.py                            # Band processing runner
    │   ├── run_electrodes.py                       # Electrode processing runner
    │   ├── vdvae_extract_features.py               # VDVAE latent extraction
    │   ├── clipvision_extract_features.py          # CLIP vision feature extraction
    │   ├── cliptext_extract_features.py            # CLIP text feature extraction
    │   └── evaluation_extract_features_from_test_images.py
    └── training/                                   # Model training and evaluation
        ├── train_regression.py                     # Ridge regression: EEG → embeddings
        ├── reconstruct_from_embeddings.py          # Image reconstruction via diffusion
        ├── evaluate_reconstruction.py              # Metrics: SSIM, LPIPS, ranking
        ├── evaluation_extract_features.py          # Feature extraction for evaluation
        ├── plot_reconstructions.py                 # Reconstruction visualization
        ├── plot_umap_CLIP.py                       # UMAP visualization of CLIP embeddings
        ├── umap_f.py                               # UMAP analysis
        ├── umap27_f.py                             # UMAP analysis variant
        └── umap3.py                                # UMAP analysis variant
```

---

## Key Technologies

| Category | Technologies |
|---|---|
| **Deep Learning** | PyTorch, HuggingFace Transformers, TorchVision, CUDA 11.8 |
| **EEG Processing** | MNE-Python, MNE-BIDS, MNE-ICALabel, SciPy |
| **Vision Models** | OpenAI CLIP, Facebook DINOv3 (ViT-B/16), VDVAE, Versatile Diffusion |
| **ML / Statistics** | scikit-learn (Ridge Regression), UMAP, torchmetrics |
| **Data Handling** | HDF5 (h5py), NumPy, Pandas, PIL/Pillow, OpenCV |
| **Distributed Training** | PyTorch DDP, Mixed Precision (AMP), Gradient Accumulation |
| **Visualization** | Matplotlib, Seaborn, Plotly, scikit-image |
| **Signal Analysis** | MATLAB (GED, filtering, topoplots) |

---

## Sub-Project Details

### 1. GED — Generalized Eigenvalue Decomposition

**Purpose:** Extracts individualized EEG frequency components per subject using classical signal decomposition.

**How it works:**
1. Convert EEG `.npy` files to MATLAB `.mat` format using `npyTomat.py`
2. Run `ged_analysis.m` in MATLAB to perform GED and identify subject-specific frequencies
3. Visualize results with topographic plots (`topoplot_eeg63.m`)

**Data source:** [OSF — EEG Dataset](https://osf.io/6sg5e/overview)

Adapted from the [GED Tutorial](https://github.com/mikexcohen/GED_tutorial) by Mike X Cohen.

---

### 2. EEG-Image Reconstruction Foundation Model

**Purpose:** A multi-stage deep learning pipeline that learns to reconstruct images from EEG signals, trained on the [AllJoined-1.6M](https://huggingface.co/datasets/Alljoined/Alljoined-1.6M) dataset (1.6 million trials across 20 subjects).

#### Pipeline Overview

```
Raw EEG (variable channels & sampling rates)
    │
    ▼  Preprocessing
Standardized EEG + paired images
    │
    ▼  Stage 1: Harmonization (Temporal Transformer)
1024-dim unified embeddings
    │
    ▼  Stage 2: EEG-ViT + Mixture of Experts
768-dim semantic embeddings
    │
    ▼  Stage 3: Contrastive Alignment with DINOv3
768-dim aligned embeddings
    │
    ▼  Stage 4: Versatile Diffusion
Reconstructed images (512×512)
```

#### Stage Details

| Stage | File | Description | Key Architecture |
|---|---|---|---|
| **Preprocessing** | `preprocessing.py`, `preprocess_all.py` | Loads raw EEG and paired images, standardizes per subject | Multiprocessing, memory optimization |
| **Stage 1** | `stage1_harmonise.py` | Maps diverse EEG setups (32–128 channels) to a unified embedding space | Spatial Channel Embedding → 6-layer Temporal Transformer → 1024-dim output |
| **Stage 2** | `stage2_eeg_vit.py` | Encodes harmonized EEG into semantically meaningful embeddings | TCN + Vision Transformer + Self-Aware MoE (4 experts, category-aware gating) → 768-dim output |
| **Stage 3** | `stage3_alignment.py` | Aligns EEG embeddings with DINOv3 image embeddings | Progressive Alignment Network + 8-head Cross-Attention + InfoNCE Contrastive Loss |
| **Stage 4** | `stage4_reconstruct.py` | Generates images from aligned embeddings | VDVAE (64×64) or Versatile Diffusion with DDIM sampling (512×512) |

#### Running the Foundation Model

```bash
# Set up environment
conda env create -f "EEG-Image Reconstruction Foundation model/environment.yaml"

# Step 1: Preprocess EEG data (subjects 1-20)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 preprocessing.py --subjects 1-20

# Step 2: Stage 1 — Harmonization
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 stage1_harmonise.py \
    --datasets /path/to/data --output_dir /path/to/output --num_subjects 20

# Step 3: Stage 2 — EEG-ViT encoding
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 stage2_eeg_vit.py --output_dir /path/to/output

# Step 4: Stage 3 — Alignment
python3 stage3_alignment.py

# Step 5: Extract aligned embeddings
python3 extract_stage3.py

# Step 6: Stage 4 — Image reconstruction
CUDA_VISIBLE_DEVICES=4 python3 stage4_reconstruct.py \
    --image_dir /path/to/images --output_path /path/to/output
```

---

### 3. Reconstruction with Bands

**Purpose:** Investigates how different EEG **frequency bands** (Delta, Theta, Alpha, Beta, Gamma) and **electrode placements** affect image reconstruction quality. Uses the [ThingsEEG2](https://osf.io/3jk45/) dataset (4 subjects).

#### Pipeline Overview

```
ThingsEEG2 raw EEG + images
    │
    ▼  Band filtering (Delta/Theta/Alpha/Beta/Gamma)
    ▼  Electrode selection (occipital, parietal, etc.)
Filtered EEG epochs
    │
    ▼  Feature Extraction (VDVAE, CLIP Vision, CLIP Text)
Image embeddings (.npy)
    │
    ▼  Ridge Regression Training
EEG → Embedding mapping
    │
    ▼  Reconstruction (Versatile Diffusion)
Reconstructed images
    │
    ▼  Evaluation (SSIM, LPIPS, ranking)
Quantitative results + UMAP visualizations
```

#### Frequency Bands

| Band | Frequency Range | Neural Association |
|---|---|---|
| Delta | 0.5–4 Hz | Deep sleep, unconscious processing |
| Theta | 5–7 Hz | Memory, spatial navigation |
| Alpha | 8–13 Hz | Relaxation, visual cortex idling |
| Beta | 14–30 Hz | Active thinking, focus |
| Gamma | 30–100 Hz | Perception, consciousness, binding |

#### Running the Bands Pipeline

```bash
# Install dependencies
pip install -r "Reconstruction with Bands/preparation/requirements.txt"

# Step 1: Prepare ThingsEEG2 data
python3 preparation/prepare_thingseeg2_data.py
python3 preparation/save_thingseeg2_images.py
python3 preparation/save_thingseeg2_concepts.py

# Step 2: Process frequency bands and electrode configurations
python3 preparation/run_bands.py -sfreq 250
python3 preparation/run_electrodes.py -sfreq 250

# Step 3: Extract features from images
python3 preparation/vdvae_extract_features.py
python3 preparation/clipvision_extract_features.py
python3 preparation/cliptext_extract_features.py
python3 preparation/evaluation_extract_features_from_test_images.py

# Step 4: Train regression models
python3 training/train_regression.py

# Step 5: Reconstruct images
python3 training/reconstruct_from_embeddings.py

# Step 6: Evaluate and visualize
python3 training/evaluate_reconstruction.py
python3 training/plot_reconstructions.py -ordered True
```

---

## Evaluation Metrics

| Metric | Description | Direction |
|---|---|---|
| **SSIM** | Structural Similarity Index | Higher is better |
| **LPIPS** | Learned Perceptual Image Patch Similarity | Lower is better |
| **Ranking Score** | Percentage of correct pairwise rankings | Higher is better |

Statistical significance is assessed via binomial tests on ranking accuracy against chance.

---

## Datasets

| Dataset | Description | Source |
|---|---|---|
| **AllJoined-1.6M** | 1.6M EEG-image trials across 20 subjects | [HuggingFace](https://huggingface.co/datasets/Alljoined/Alljoined-1.6M) |
| **ThingsEEG2** | EEG responses to object images, 4 subjects | [OSF](https://osf.io/3jk45/) |
| **ds005106** | Infant EEG dataset | [OpenNeuro](https://openneuro.org/datasets/ds005106/versions/1.5.0) |
| **GED EEG Data** | Resting-state EEG for frequency analysis | [OSF](https://osf.io/6sg5e/) |

---

## Pretrained Models

| Model | Purpose | Source |
|---|---|---|
| **DINOv2 ViT-B/16** | Image embeddings for alignment | [Facebook Research](https://github.com/facebookresearch/dinov2) |
| **CLIP** | Vision and text embeddings | [OpenAI](https://github.com/openai/CLIP) |
| **VDVAE** | Variational autoencoder for latent codes | [OpenAI](https://github.com/openai/vdvae) |
| **Versatile Diffusion** | Diffusion-based image generation | [HuggingFace](https://huggingface.co/shi-labs/versatile-diffusion) |

---

## Acknowledgments

This project builds upon and adapts code from:

- [GED Tutorial](https://github.com/mikexcohen/GED_tutorial) — Mike X Cohen
- [Perceptogram](https://github.com/desa-lab/Perceptogram) — Fei et al., 2024
- [AllJoined-1.6M](https://github.com/Alljoined/Alljoined-1.6M) — Xu et al., 2025
- [DINOv2](https://github.com/facebookresearch/dinov2) — Facebook Research
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion) — CompVis

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
