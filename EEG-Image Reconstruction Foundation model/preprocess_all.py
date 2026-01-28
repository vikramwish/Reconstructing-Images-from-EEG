import os
from concurrent.futures import ThreadPoolExecutor
import gc

# Performance optimizations
os.environ['MNE_USE_NUMBA'] = 'true'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['NUMBA_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

from concurrent.futures import ThreadPoolExecutor
import threading
import mne
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import dataclasses
import json
import warnings
from typing import List, Dict, Optional, Tuple
import h5py
import time
import os
import re
import argparse
from all_categories import get_categories

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# Setup logging without timestamp
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GlobalPreprocessingConfig:
    """Universal preprocessing parameters for AllJoined"""
    base_dataset_dir: Path = Path("/raid/datasets/tanaya/fm/datasets/")
    output_dir: Path = Path("/raid/datasets/tanaya/fm/preprocess/")

    # AllJoined-specific parameters
    sfreq: int = 250  # Target sampling rate
    l_freq: float = 0.5  # High-pass filter (Hz)
    h_freq: float = 100.0  # Low-pass filter (Hz)
    notch_freqs: List[float] = dataclasses.field(default_factory=lambda: [60.0])  # US power line

    # Phase 1: Time domain data with image mapping
    apply_normalization: bool = False
    save_raw_microvolts: bool = True


@dataclasses.dataclass
class AllJoinedConfig:
    """AllJoined-specific configuration"""
    name: str = "alljoined"
    raw_dir: Optional[Path] = None
    image_dir: Optional[Path] = None

    # Timing parameters
    t_min: float = -0.2
    t_max: float = 1.0
    baseline: Tuple[Optional[float], float] = (None, 0)

    # Session and block organization
    expected_sessions: List[int] = dataclasses.field(default_factory=lambda: [1, 2, 3, 4])
    expected_blocks: List[int] = dataclasses.field(default_factory=lambda: list(range(1, 21)))

    # AllJoined-specific bad channel prefixes
    bad_channel_prefixes: List[str] = dataclasses.field(default_factory=lambda: [
        "TimestampS", "TimestampMs", "OrTimestampS", "OrTimestampMs",
        "Counter", "Interpolated", "RawCq", "Battery", "BatteryPercent",
        "FwBufferSize", "FwClockTime", "MarkerHardware", "HighBitFlex",
        "SaturationFlag", "CQ", "EQ", "MOT"
    ])

    # Rejection criteria
    rejection_criteria: Dict[str, float] = dataclasses.field(default_factory=lambda: {'eeg': 200e-6})
    use_rejection: bool = False

    def __post_init__(self):
        if self.raw_dir is None:
            self.raw_dir = Path("/raid/datasets/tanaya/fm/datasets/alljoined/raw_eeg/")
        if self.image_dir is None:
            self.image_dir = Path("/raid/datasets/tanaya/fm/datasets/alljoined/images/")

class AllJoinedPreprocessor:
    """AllJoined dataset preprocessor with proper image mapping"""

    def __init__(self, global_config: GlobalPreprocessingConfig, alljoined_config: AllJoinedConfig):
        self.config = global_config
        self.alljoined_config = alljoined_config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Load the category hierarchy from all_categories.py
        self.categories_lib = get_categories()

        logger.info("AllJoined Preprocessor initialized with image mapping")
        logger.info(f"Raw data dir: {self.alljoined_config.raw_dir}")
        logger.info(f"Image dir: {self.alljoined_config.image_dir}")
        logger.info(f"Output dir: {self.config.output_dir}")
        logger.info(f"Loaded {len(self.categories_lib)} category hierarchies")

        # Debug category loading
        logger.info("Available super-categories:")
        for cat_name, items in self.categories_lib.items():
            logger.info(f"  {cat_name}: {len(items)} items")

        #patch_alljoined_preprocessor(self)

    # Key optimization: Parallel EDF loading with pre-caching
    def load_all_blocks_parallel(self, subject_id: int, session: int, blocks: List[int]) -> List[mne.io.Raw]:
        """Load all blocks for a session in parallel with caching"""

        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=20) as executor:  # More workers for I/O
            futures = []
            for block in blocks:
                future = executor.submit(self._load_single_block_cached,
                                         subject_id, session, block)
                futures.append((block, future))

            # Collect results
            raws = []
            for block, future in sorted(futures, key=lambda x: x[0]):
                raw = future.result()
                if raw is not None:
                    raws.append(raw)

        return raws

    def _load_single_block_cached(self, subject_id: int, session: int, block: int) -> Optional[mne.io.Raw]:
        """Load single block with caching"""
        cache_key = f"sub{subject_id:02d}_sess{session}_block{block}"

        # Check memory cache first
        if hasattr(self, '_raw_cache') and cache_key in self._raw_cache:
            return self._raw_cache[cache_key]

        # Load from disk
        subject_dir = self.raw_dir / f"sub-{subject_id:02d}"
        session_dir = subject_dir / f"session_{session:02d}"
        block_dir = session_dir / f"block_{block:02d}"

        edf_files = list(block_dir.glob("*.edf"))
        if not edf_files:
            return None

        # Load with minimal preprocessing
        raw = mne.io.read_raw_edf(edf_files[0], preload=False, verbose=False)

        # Pick channels first to reduce data size
        eeg_channels = self._get_electrode_channels(raw)
        raw.pick(eeg_channels)

        # Now load data
        raw.load_data()

        # Cache if enabled
        if hasattr(self, '_enable_cache') and self._enable_cache:
            if not hasattr(self, '_raw_cache'):
                self._raw_cache = {}
            self._raw_cache[cache_key] = raw

        return raw

    def _load_experiment_metadata_categories(self, subject_id: int) -> pd.DataFrame:
        """Load and process the experiment_metadata.parquet file for a specific subject"""

    
        logger.info(f"Loading subject-specific experiment metadata from: {metadata_path}")
        metadata_df = pd.read_parquet(metadata_path)

        logger.info(f"Loaded {len(metadata_df)} trials from subject {subject_id:02d} experiment metadata")
        logger.info(f"Columns available: {list(metadata_df.columns)}")

        # Filter for stim_train and stim_test partitions
        if 'partition' in metadata_df.columns:
            stimulus_trials = metadata_df[metadata_df['partition'].isin(['stim_train', 'stim_test'])].copy()
            logger.info(f"Stimulus trials (train+test, excluding oddball): {len(stimulus_trials)}")
        else:
            stimulus_trials = metadata_df.copy()
            logger.info(f"Using all trials: {len(stimulus_trials)}")

        # Reset index
        stimulus_trials = stimulus_trials.reset_index(drop=True)

        # Add derived columns if missing
        if 'image_path_full' not in stimulus_trials.columns and 'fname' in stimulus_trials.columns:
            stimulus_trials['image_path_full'] = stimulus_trials['fname'].apply(
                lambda x: self.alljoined_config.image_dir / x if pd.notna(x) else None
            )

        if 'image_exists' not in stimulus_trials.columns and 'image_path_full' in stimulus_trials.columns:
            stimulus_trials['image_exists'] = stimulus_trials['image_path_full'].apply(
                lambda x: Path(x).exists() if x is not None else False
            )

        # Add session assignments if missing
        if 'session' not in stimulus_trials.columns:
            total_trials = len(stimulus_trials)
            trials_per_session = total_trials // 4
            stimulus_trials['session'] = 1
            stimulus_trials.loc[trials_per_session:2 * trials_per_session, 'session'] = 2
            stimulus_trials.loc[2 * trials_per_session:3 * trials_per_session, 'session'] = 3
            stimulus_trials.loc[3 * trials_per_session:, 'session'] = 4

        return stimulus_trials

    def _get_electrode_channels(self, raw: mne.io.Raw) -> List[str]:
        """Return genuine EEG channel names"""
        bad_prefixes = {
            "TimestampS", "TimestampMs", "OrTimestampS", "OrTimestampMs",
            "Counter", "Interpolated", "RawCq", "Battery", "BatteryPercent",
            "FwBufferSize", "FwClockTime", "MarkerHardware", "HighBitFlex",
            "SaturationFlag", "CQ", "EQ", "MOT",
        }
        return [
            ch for ch in raw.ch_names
            if not any(ch.startswith(p) for p in bad_prefixes)
        ]


    def _epoching_by_sessions(self, subject_id: int, d_config: AllJoinedConfig) -> List[mne.Epochs]:
        """Parallel session processing - much faster"""
        from functools import partial

        # Create partial function with fixed arguments
        process_func = partial(
            self._process_single_session,
            blocks=d_config.expected_blocks,
            subject_id=subject_id,
            d_config=d_config
        )

        # Process sessions in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            all_epochs = list(executor.map(process_func, d_config.expected_sessions))

        # Filter out None values and log results
        valid_epochs = []
        for i, epochs in enumerate(all_epochs):
            if epochs is not None:
                valid_epochs.append(epochs)
                logger.info(f"Session {d_config.expected_sessions[i]}: {len(epochs)} epochs loaded")
            else:
                logger.warning(f"Session {d_config.expected_sessions[i]}: No epochs")

        return valid_epochs

    def _process_single_session(self, session: int, blocks: List[int], subject_id: int, d_config: AllJoinedConfig):
        """Process one session with multiple blocks - optimized sequential loading"""
        logger.info(f"Loading session {session}...")
        subject_dir = self.alljoined_config.raw_dir / f"sub-{subject_id:02d}"
        session_dir = subject_dir / f"session_{session:02d}"

        if not session_dir.exists():
            logger.warning(f"Session directory not found: {session_dir}")
            return None

        from concurrent.futures import ThreadPoolExecutor


       
        raws = []
        for i, block in enumerate(blocks, 1):
            if i % 5 == 0:  # Progress every 5 blocks
                logger.info(f"  Loading block {i}/{len(blocks)}")
        for block in blocks:
            block_dir = session_dir / f"block_{block:02d}"
            if not block_dir.exists():
                continue

            edf_files = list(block_dir.glob("*.edf"))
            if not edf_files:
                continue

            try:
                #raw = mne.io.read_raw_edf(edf_files[0], preload=True, verbose=False)
                raw = mne.io.read_raw_edf(edf_files[0], preload=False, verbose=False) #to make faster
                if "Afz" in raw.ch_names:
                    raw.rename_channels({"Afz": "AFz"})

                eeg_channels = self._get_electrode_channels(raw)
                raw.pick(eeg_channels)
                #raw.pick(eeg_channels, ordered=False)
                raw.load_data()
                raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
                raws.append(raw)
            except Exception as e:
                logger.warning(f"Failed to load session {session}, block {block}: {e}")
                continue

        if not raws:
            return None

        # Apply filtering per session
        for b_idx, raw in enumerate(raws, start=1):
            annotations = raw.annotations
            if self.config.l_freq is not None or self.config.h_freq is not None:
                raw.filter(self.config.l_freq, self.config.h_freq, verbose=False, n_jobs=4)
            if self.config.notch_freqs:
                raw.notch_filter(self.config.notch_freqs, verbose=False, n_jobs=4)

            # Make annotations unique
            raw.set_annotations(mne.Annotations(
                onset=annotations.onset,
                duration=annotations.duration,
                description=[f"session_{session},block_{b_idx},{d}" for d in annotations.description],
                orig_time=annotations.orig_time,
            ))

        raw_concat = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]
        events, event_id = mne.events_from_annotations(raw_concat, regexp=".*stim", verbose=False)

        if len(events) == 0:
            logger.warning(f"No events found in session {session}")
            return None

        epochs = mne.Epochs(
            raw_concat, events, event_id=event_id,
            tmin=d_config.t_min, tmax=d_config.t_max, baseline=d_config.baseline,
            preload=True, reject=None, event_repeated="drop", verbose=False
        )

        if epochs.info["sfreq"] != self.config.sfreq:
            epochs.resample(self.config.sfreq, verbose=False, n_jobs=4)

        logger.info(f"Session {session}: {len(epochs)} epochs created")
        del raw
        gc.collect()

        return epochs

    def _create_trial_image_mapping(self, epochs_list: List[mne.Epochs],
                                    stim_order: pd.DataFrame) -> pd.DataFrame:
        """Align EEG triggers with subject-specific stim_order"""
        trial_mapping = []

        for sess, epochs in enumerate(epochs_list, start=1):
            events = epochs.events
            code_to_desc = {v: k for k, v in epochs.event_id.items()}
            # Decode trigger values from event descriptions
            trigger_vals = np.array([int(code_to_desc[e].split(",")[3]) for e in events[:, 2]])

            # Subject-specific stim_order for this session
            session_df = stim_order[stim_order["session"] == sess].copy()
            session_df["image_index"] = session_df["image_path"].str[-9:-4].astype(int)

            img_idx_arr = session_df["image_index"].to_numpy()

            indices, ptr = [], 0
            for val in trigger_vals:
                while ptr < len(img_idx_arr) and img_idx_arr[ptr] != val:
                    ptr += 1
                if ptr == len(img_idx_arr):
                    raise ValueError(f"Session {sess}: trigger sequence not in stim_order")
                indices.append(session_df.index[ptr])
                ptr += 1

            kept_metadata = session_df.loc[indices].reset_index(drop=True)

            for trial_idx, (onset_sample, _, event_code) in enumerate(events):
                onset_time = onset_sample / epochs.info["sfreq"]
                metadata_row = kept_metadata.iloc[trial_idx]

                trial_mapping.append({
                    "subject": f"sub-{self._current_subject_id:02d}",
                    "session": sess,
                    "trial_index": trial_idx,
                    "onset_sample": int(onset_sample),
                    "onset_time": float(onset_time),
                    "event_code": int(event_code),
                    "image_index": metadata_row["category_img_num"],
                    "image_category": metadata_row["category_name"],
                    "image_filename": Path(metadata_row["image_path"]).name,
                    "image_path": str(metadata_row["image_path"]),
                    # FIXED: Use scalar access instead of list access
                    "super_category": metadata_row.get("super_category", "unknown"),
                    "super_category_id": metadata_row.get("super_category_id", 0),
                    #"minor_category": metadata_row.get("minor_category", "unknown"),
                    #"minor_category_id": metadata_row.get("minor_category_id", 0),
                    "mapped": True,
                })

        return pd.DataFrame(trial_mapping)

    def _get_super_category(self, category_name: str) -> str:
        """Map individual item to super-category"""
        if not category_name or pd.isna(category_name) or category_name == 'unknown':
            return 'unknown'

        # Ensure we're working with a clean string
        category_name = str(category_name).strip().lower()

        # Direct lookup in hierarchy
        for super_cat, word_list in self.categories_lib.items():
            # Convert all words in the list to lowercase for comparison
            word_list_lower = [word.lower() for word in word_list]
            if category_name in word_list_lower:
                return super_cat

        # If not found in any category, log it for debugging
        logger.warning(f"Category '{category_name}' not found in any super-category, assigning to 'unknown'")
        return 'unknown'

    def _get_super_category_id(self, category_name: str) -> int:
        """Get numeric ID for super-category based on AllJoined hierarchy"""
        super_cat = self._get_super_category(category_name)
        category_ids = {
            'animals': 1,
            'foods_and_plants': 2,
            'household_items_and_furniture': 3,
            'tools': 4,
            'vehicles': 5,
            'body_parts_and_apparel': 6,
            'toys_and_games': 7
        }

        # Since we filtered out oddball partition, all categories should be mappable
        # If we get 'unknown', it means there's a missing item in all_categories.py
        if super_cat == 'unknown':
            logger.error(
                f"Category '{category_name}' not found in all_categories.py - this shouldn't happen after filtering!")
            return 0  # Fallback, but investigate why this occurred

        return category_ids.get(super_cat, 0)

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    def _save_processed_data(self, subject_id: int, epochs: mne.Epochs,
                             trial_mapping: pd.DataFrame, processing_metadata: Dict):
        """Save processed EEG data and trial-image mapping"""

        subject_str = f"sub-{subject_id:02d}"
        output_dir = self.config.output_dir / "alljoined" / subject_str
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get EEG data
        eeg_data = epochs.get_data()  # Shape: (n_trials, n_channels, n_times)

        # Convert to microvolts if needed
        if self.config.save_raw_microvolts:
            eeg_data = eeg_data * 1e6

        # Save EEG data to HDF5
        output_file = output_dir / f"{subject_str}_processed.h5"

        with h5py.File(output_file, 'w') as f:
            # Save EEG data with compression
            f.create_dataset('eeg_data', data=eeg_data, compression='gzip')
            f.create_dataset('channel_names',
                             data=[ch.encode('utf-8') for ch in epochs.ch_names])
            f.create_dataset('times', data=epochs.times)

            # Save essential metadata as attributes
            f.attrs['subject_id'] = subject_str
            f.attrs['dataset_name'] = 'alljoined'
            f.attrs['n_trials'] = eeg_data.shape[0]
            f.attrs['n_channels'] = eeg_data.shape[1]
            f.attrs['n_times'] = eeg_data.shape[2]
            f.attrs['sfreq'] = epochs.info['sfreq']
            f.attrs['tmin'] = epochs.tmin
            f.attrs['tmax'] = epochs.tmax
            f.attrs['processing_time'] = processing_metadata['processing_time']

        # Save trial-to-image mapping as parquet (universal format)
        mapping_file = output_dir / "mapping_trial.parquet"
        trial_mapping.to_parquet(mapping_file)

 
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            serializable_metadata = self._convert_numpy_types(processing_metadata)
            json.dump(serializable_metadata, f, indent=2)

        # Log file sizes
        eeg_size = output_file.stat().st_size / 1e6
        mapping_size = mapping_file.stat().st_size / 1e3

        logger.info(f"Saved data files:")
        logger.info(f"  EEG data: {output_file.name} ({eeg_size:.1f} MB)")
        logger.info(f"  Image mapping: {mapping_file.name} ({mapping_size:.1f} KB)")
        logger.info(f"  Metadata: {metadata_file.name}")

        return output_file

    def process_subject(self, subject_id: int) -> Dict:
        """Process single subject with image mapping"""

        start_time = time.time()
        subject_str = f"sub-{subject_id:02d}"
        self._current_subject_id = subject_id

        logger.info("=" * 60)
        logger.info(f"Processing AllJoined Subject {subject_str}")
        logger.info("=" * 60)

        try:

            logger.info("Step 1: Loading subject-specific experiment metadata...")
            #experiment_metadata = self._load_experiment_metadata_categories(subject_id)  # Pass subject_id
            stim_order = self._load_experiment_metadata_categories(subject_id)  # per-subject stim_order

      
            logger.info("Step 2: Loading EEG data from all sessions...")
            #all_epochs_sessions = self._epoching_by_sessions(subject_id, self.alljoined_config)
            all_epochs_sessions = self._epoching_by_sessions(subject_id, self.alljoined_config)

            if not all_epochs_sessions:
                raise ValueError(f"No epochs created for subject {subject_id}")

           
            logger.info("Step 3: Creating trial-to-image mapping...")
            #trial_mapping = self._create_trial_image_mapping(all_epochs_sessions, experiment_metadata)
            trial_mapping = self._create_trial_image_mapping(all_epochs_sessions, stim_order)


            logger.info("Step 4: Concatenating epochs from all sessions...")
            epochs = mne.concatenate_epochs(all_epochs_sessions)
            logger.info(f"Total epochs after concatenation: {len(epochs)}")

            # Prepare processing metadata
            processing_time = time.time() - start_time
            eeg_data = epochs.get_data()

            processing_metadata = {
                'subject_id': subject_str,
                'dataset_name': 'alljoined',
                'processing_time': processing_time,
                'n_trials_total': len(epochs),
                'n_trials_mapped': int(trial_mapping['mapped'].sum()),
                'mapping_success_rate': float(trial_mapping['mapped'].mean()),
                'n_channels': len(epochs.ch_names),
                'n_timepoints': len(epochs.times),
                'channel_names': epochs.ch_names,
                'sampling_rate': float(epochs.info['sfreq']),
                'time_window': [float(epochs.tmin), float(epochs.tmax)],
                'amplitude_range': [float(np.min(eeg_data)), float(np.max(eeg_data))],
                # FIXED: Use len() instead of nunique() to avoid Series issues
                'unique_images': len(trial_mapping['image_index'].unique()),
                'unique_categories': len(trial_mapping['super_category'].unique()),
                #'categories_found': trial_mapping['minor_category'].dropna().unique().tolist()
            }

            # Save processed data
            logger.info("Step 5: Saving processed data...")
            output_file = self._save_processed_data(subject_id, epochs, trial_mapping, processing_metadata)

            # Success summary
            logger.info("SUCCESS! Processing completed")
            logger.info(f"  Trials: {processing_metadata['n_trials_total']}")
            logger.info(f"  Mapped: {processing_metadata['n_trials_mapped']}")
            logger.info(f"  Success rate: {processing_metadata['mapping_success_rate']:.1%}")
            logger.info(f"  Categories: {processing_metadata['unique_categories']}")
            logger.info(f"  Processing time: {processing_time:.1f}s")

            return {
                'status': 'success',
                'subject_id': subject_str,
                'output_file': str(output_file),
                'shape': eeg_data.shape,
                'n_trials': processing_metadata['n_trials_total'],
                'n_channels': processing_metadata['n_channels'],
                'trials_mapped': processing_metadata['n_trials_mapped'],
                'mapping_success_rate': processing_metadata['mapping_success_rate'],
                'unique_images': processing_metadata['unique_images'],
                'unique_categories': processing_metadata['unique_categories'],
                #'categories_found': processing_metadata['categories_found'],
                'amplitude_range': processing_metadata['amplitude_range'],
                'processing_time': processing_time,
                'image_mapping_created': True
            }

        except Exception as e:
            error_time = time.time() - start_time
            import traceback
            logger.error(f"ERROR processing {subject_str}: {e}")
            logger.error(f"Error traceback: {traceback.format_exc()}")

            return {
                'status': 'error',
                'subject_id': subject_str,
                'error': str(e),
                'processing_time': error_time,
                'image_mapping_created': False
            }


def main():
    """Main function with command line argument support"""

    parser = argparse.ArgumentParser(description='AllJoined Preprocessing with Image Mapping')
    parser.add_argument('--subject', type=int, default=1, help='Subject ID to process')
    parser.add_argument('--output-dir', type=str, default='/raid/datasets/tanaya/fm/prep/',
                        help='Output directory')
    args = parser.parse_args()

    # Configuration
    global_config = GlobalPreprocessingConfig(
        base_dataset_dir=Path("/raid/datasets/tanaya/fm/datasets/"),
        output_dir=Path(args.output_dir),
        sfreq=250,
        l_freq=0.5,
        h_freq=100.0,
        notch_freqs=[60.0],
        apply_normalization=False,
        save_raw_microvolts=True
    )

    alljoined_config = AllJoinedConfig()

    # Create preprocessor
    preprocessor = AllJoinedPreprocessor(global_config, alljoined_config)

    subject_to_process = args.subject

    logger.info("AllJoined Preprocessing with Image Mapping")
    logger.info(f"Target: (n_trials, n_channels, {int((1.0 - (-0.2)) * 250)}_timepoints)")
    logger.info(f"Subject: {subject_to_process:02d}")
    logger.info(f"Output: {global_config.output_dir}")

    try:
        result = preprocessor.process_subject(subject_to_process)

        if result['status'] == 'success':
            print("\n" + "=" * 60)
            print("PREPROCESSING SUCCESSFUL!")
            print("=" * 60)
            print(f"Subject: {result['subject_id']}")
            print(f"Shape: {result['shape']}")
            print(f"Trials: {result['n_trials']}")
            print(f"Channels: {result['n_channels']}")
            print(f"Mapped trials: {result['trials_mapped']}")
            print(f"Mapping success: {result['mapping_success_rate']:.1%}")
            print(f"Unique images: {result['unique_images']}")
            print(f"Categories: {result['unique_categories']}")
            #print(f"Categories found: {result['categories_found']}")
            print(f"Amplitude range: {result['amplitude_range'][0]:.1e} to {result['amplitude_range'][1]:.1e}")
            print(f"Processing time: {result['processing_time']:.1f}s")
            print(f"Output: {result['output_file']}")


        else:
            print("\n" + "=" * 60)
            print("PREPROCESSING FAILED!")
            print("=" * 60)
            print(f"Subject: {result['subject_id']}")
            print(f"Error: {result['error']}")
            print(f"Processing time: {result['processing_time']:.1f}s")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        print(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()