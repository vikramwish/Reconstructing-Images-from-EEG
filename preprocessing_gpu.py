
import multiprocessing as mp
from multiprocessing import Pool, get_context
import argparse
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import sys
import os
import traceback
import gc
import psutil
from functools import partial

#  Set multiprocessing start method early
try: 
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Performance optimizations
os.environ['MNE_USE_NUMBA'] = 'true'
os.environ['NUMEXPR_MAX_THREADS'] = '40'  # Use all cores for NumExpr
os.environ['OMP_NUM_THREADS'] = '4'       # Increased per process
os.environ['NUMBA_NUM_THREADS'] = '4'     # Increased per process
os.environ['MKL_NUM_THREADS'] = '4'       # Increased per process

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parallel_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_subjects(subjects_str: str) -> List[int]:
    """Parse subject specification like '1-20' or '1,2,3'"""
    if not subjects_str:
        return list(range(1, 21))  # Default to all 20 AllJoined subjects

    subjects = []
    for part in subjects_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            subjects.extend(range(start, end + 1))
        else:
            subjects.append(int(part))

    return sorted(list(set(subjects)))


def setup_gpu_environment(gpu_id: Optional[int]):
    """Setup GPU environment for a process"""
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

        # Try to import torch and set device
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(0)  # Since we only see one GPU
                # Clear GPU cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass


def process_single_subject(args: Tuple) -> Dict:
    """
    Process a single subject - optimized worker function
    This runs in a separate process
    """
    import traceback

    subject_id, config_dict, gpu_id, process_id = args

    # Setup GPU for this process
    setup_gpu_environment(gpu_id)

    # Import heavy libraries only in worker process to save memory
    #from prep_alljoined import AllJoinedPreprocessor, AllJoinedConfig, GlobalPreprocessingConfig
    from preprocess_all import AllJoinedPreprocessor, AllJoinedConfig, GlobalPreprocessingConfig
    import warnings
    warnings.filterwarnings('ignore')

    start_time = time.time()
    subject_str = f"sub-{subject_id:02d}"

    try:
        # Check if already processed
        output_dir = Path(config_dict['output_dir']) / 'alljoined' / subject_str
        output_file = output_dir / f"{subject_str}_preprocessed.h5"

        if output_file.exists() and not config_dict.get('force', False):
            logger.info(f"[Process {process_id}] Subject {subject_id:02d} already processed, skipping")
            return {
                'status': 'skipped',
                'subject_id': subject_str,
                'processing_time': 0,
                'gpu_id': gpu_id,
                'process_id': process_id
            }

        # Recontruct configurations
        global_config = GlobalPreprocessingConfig(
            base_dataset_dir=Path(config_dict['base_dataset_dir']),
            output_dir=Path(config_dict['output_dir']),
            sfreq=config_dict['sfreq'],
            l_freq=config_dict['l_freq'],
            h_freq=config_dict['h_freq'],
            notch_freqs=config_dict['notch_freqs'],
            apply_normalization=config_dict['apply_normalization'],
            save_raw_microvolts=config_dict['save_raw_microvolts']
        )

        alljoined_config = AllJoinedConfig(
            use_rejection=config_dict.get('use_rejection', False),
            rejection_criteria={'eeg': config_dict.get('rejection_threshold', 200e-6)}
        )

        # Create preprocessor
        preprocessor = AllJoinedPreprocessor(global_config, alljoined_config)

        # Process subject
        logger.info(f"[Process {process_id}, GPU {gpu_id}] Processing subject {subject_id:02d}")
        result = preprocessor.process_subject_with_amp(subject_id)

        # Add process metadata
        result['gpu_id'] = gpu_id
        result['process_id'] = process_id
        result['processing_time'] = time.time() - start_time

        # Force garbage collection
        gc.collect()

        return result

    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"[Process {process_id}] Error processing subject {subject_id:02d}: {e}")
        logger.error(traceback.format_exc())

        return {
            'status': 'failed',
            'subject_id': subject_str,
            'error': str(e),
            'processing_time': error_time,
            'gpu_id': gpu_id,
            'process_id': process_id
        }


def estimate_processing_time(num_subjects: int, num_processes: int, use_gpu: bool = False) -> str:
    """Estimate total processing time"""

    if use_gpu:
        minutes_per_subject = 3.0  # GPU with optimizations
    else:
        minutes_per_subject = 10.0  # CPU baseline

    total_minutes = (num_subjects * minutes_per_subject) / num_processes

    if total_minutes < 60:
        return f"{total_minutes:.1f} minutes"
    else:
        return f"{total_minutes / 60:.1f} hours"


def get_available_gpus() -> List[int]:
    """Get list of available GPUs"""
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_devices:
        return [int(d) for d in cuda_devices.split(',') if d.strip().isdigit()]

    # Try to detect with torch
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except ImportError:
        pass

    return []


def get_optimal_process_count(num_subjects: int, use_gpu: bool, num_gpus: int) -> int:
    """Calculate optimal number of processes"""
    if use_gpu and num_gpus > 0:
        # For GPU: 1-2 processes per GPU is usually optimal
        # More processes can cause GPU memory fragmentation
        return min(num_gpus * 1, num_subjects)  # 1 process per GPU
    else:
        # For CPU: use most cores but leave some for system
        cpu_count = psutil.cpu_count(logical=False) or 4
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)

        # Estimate 4GB per process for safety
        max_memory_processes = int(memory_gb // 4)
        max_cpu_processes = max(1, cpu_count - 2)  # Leave 2 cores for system

        return min(max_memory_processes, max_cpu_processes, num_subjects, 20)


def serialize_config(global_config, alljoined_config, args) -> Dict:
    """Serialize configurations for multiprocessing"""
    return {
        'base_dataset_dir': str(global_config.base_dataset_dir),
        'output_dir': str(global_config.output_dir),
        'sfreq': global_config.sfreq,
        'l_freq': global_config.l_freq,
        'h_freq': global_config.h_freq,
        'notch_freqs': global_config.notch_freqs,
        'apply_normalization': global_config.apply_normalization,
        'save_raw_microvolts': global_config.save_raw_microvolts,
        'use_rejection': args.rejection,
        'rejection_threshold': args.rejection_threshold,
        'force': args.force
    }


def main():
    parser = argparse.ArgumentParser(
        description='Optimized Multi-GPU AllJoined Preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Subject selection
    parser.add_argument('--subjects', type=str, required=True,
                        help='Subjects: "1", "1,2,3", or "1-20"')
    parser.add_argument('--dataset', type=str, default='alljoined',
                        help='Dataset name (default: alljoined)')

    # Processing options
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of parallel processes (auto-detect if not specified)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Enable GPU acceleration')
    parser.add_argument('--processes-per-gpu', type=int, default=1,
                        help='Processes per GPU (default: 1)')

    # Data options
    parser.add_argument('--output-dir', type=str,
                        default='/raid/datasets/tanaya/fm/prep/',
                        help='Output directory')
    parser.add_argument('--rejection', action='store_true',
                        help='Enable artifact rejection')
    parser.add_argument('--rejection-threshold', type=float, default=200e-6,
                        help='Rejection threshold in V (default: 200e-6)')

    # Other options
    parser.add_argument('--dry-run', action='store_true',
                        help='Show plan without processing')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if output exists')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Parse subjects
    subjects_to_process = parse_subjects(args.subjects)
    logger.info(f"Subjects to process: {subjects_to_process}")

    # Configuration
    #from prep_alljoined import GlobalPreprocessingConfig, AllJoinedConfig
    from all import GlobalPreprocessingConfig, AllJoinedConfig

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

    alljoined_config = AllJoinedConfig(
        use_rejection=args.rejection,
        rejection_criteria={'eeg': args.rejection_threshold} if args.rejection else None
    )

    # Detect GPUs and determine process count
    available_gpus = get_available_gpus() if args.use_gpu else []
    use_gpu = len(available_gpus) > 0 and args.use_gpu

    if args.processes is None:
        optimal_processes = get_optimal_process_count(
            len(subjects_to_process),
            use_gpu,
            len(available_gpus)
        )
    else:
        optimal_processes = min(args.processes, len(subjects_to_process))

    # Display configuration
    print("\n" + "=" * 70)
    print("OPTIMIZED ALLJOINED MULTI-GPU PREPROCESSING")
    print("=" * 70)
    print(f"Subjects: {subjects_to_process}")
    print(f"Total subjects: {len(subjects_to_process)}")
    print(f"Processing mode: {'GPU-accelerated' if use_gpu else 'CPU-only'}")
    if use_gpu:
        print(f"Available GPUs: {available_gpus}")
        print(f"GPU assignment: Round-robin across {len(available_gpus)} GPUs")
    print(f"Parallel processes: {optimal_processes}")
    print(f"Estimated time: {estimate_processing_time(len(subjects_to_process), optimal_processes, use_gpu)}")
    print(f"Output directory: {global_config.output_dir}")
    print(f"Rejection: {'ENABLED' if args.rejection else 'DISABLED'}")
    if args.rejection:
        print(f"  Threshold: {args.rejection_threshold:.0e} µV")
    print("=" * 70)

    if args.dry_run:
        print("\nDRY RUN - Processing plan:")
        print("-" * 40)
        for i, subject in enumerate(subjects_to_process):
            gpu_info = ""
            if use_gpu:
                gpu_id = available_gpus[i % len(available_gpus)]
                gpu_info = f" [GPU {gpu_id}]"
            print(f"{i + 1:3d}. Subject {subject:02d}{gpu_info}")
        print(f"\nTotal: {len(subjects_to_process)} subjects")
        print(f"Parallel batches: {(len(subjects_to_process) + optimal_processes - 1) // optimal_processes}")
        return

    # Confirm processing for large jobs
    if len(subjects_to_process) > 10 and not args.force:
        response = input(f"\nProcess {len(subjects_to_process)} subjects? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled")
            return

    # Serialize configuration
    config_dict = serialize_config(global_config, alljoined_config, args)

    # Create job arguments with GPU cycling
    job_args = []
    for i, subject_id in enumerate(subjects_to_process):
        gpu_id = available_gpus[i % len(available_gpus)] if use_gpu else None
        process_id = i % optimal_processes
        job_args.append((subject_id, config_dict, gpu_id, process_id))

    print(f"\nStarting processing with {optimal_processes} parallel processes...")
    print("Note: Initial startup may take a moment as processes initialize\n")

    start_time = time.time()
    results = []

    try:
        if optimal_processes == 1:
            # Sequential processing
            logger.info("Sequential processing mode")
            for i, job_arg in enumerate(job_args):
                subject_id = job_arg[0]
                print(f"[{i + 1}/{len(job_args)}] Processing subject {subject_id:02d}...")
                result = process_single_subject(job_arg)
                results.append(result)

                # Display result
                if result['status'] == 'success':
                    print(f"  ✓ Success: {result.get('n_trials', 'N/A')} trials | "
                          f"{result['processing_time']:.1f}s")
                elif result['status'] == 'skipped':
                    print(f"  ⊙ Skipped (already processed)")
                else:
                    print(f"  ✗ Failed: {result.get('error', 'Unknown error')[:50]}")

        else:
            # Parallel processing with progress tracking
            logger.info(f"Parallel processing with {optimal_processes} processes")

            # Use spawn context for GPU compatibility
            ctx = get_context('spawn')

            with ctx.Pool(processes=optimal_processes) as pool:
                # Submit all jobs
                async_results = [pool.apply_async(process_single_subject, (job_arg,))
                                 for job_arg in job_args]

                # Collect results with progress bar
                for i, async_result in enumerate(async_results):
                    try:
                        result = async_result.get(timeout=3600)  # 10 min timeout per subject
                        results.append(result)

                        subject_id = job_args[i][0]
                        status_icon = {'success': '✓', 'skipped': '⊙', 'failed': '✗'}.get(
                            result['status'], '?'
                        )

                        print(f"[{i + 1}/{len(job_args)}] Subject {subject_id:02d}: {status_icon} "
                              f"({result['processing_time']:.1f}s)")

                    except Exception as e:
                        subject_id = job_args[i][0]
                        print(f"[{i + 1}/{len(job_args)}] Subject {subject_id:02d}: ✗ Timeout/Error")
                        results.append({
                            'status': 'failed',
                            'subject_id': f"sub-{subject_id:02d}",
                            'error': str(e),
                            'processing_time': 0
                        })

        # Calculate statistics
        total_time = time.time() - start_time
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        skipped = [r for r in results if r['status'] == 'skipped']

        # Results summary
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Successful: {len(successful)}/{len(results)}")
        print(f"Failed: {len(failed)}/{len(results)}")
        print(f"Skipped: {len(skipped)}/{len(results)}")
        print(f"Total time: {total_time / 60:.1f} minutes")

        if successful:
            processing_times = [r['processing_time'] for r in successful]
            print(f"Average time per subject: {np.mean(processing_times):.1f}s")
            print(f"Time range: {min(processing_times):.1f}s - {max(processing_times):.1f}s")

            # Calculate parallel efficiency
            if optimal_processes > 1:
                theoretical_time = sum(processing_times)
                efficiency = (theoretical_time / total_time) / optimal_processes * 100
                print(f"Parallel efficiency: {efficiency:.1f}%")

        # Log failed subjects
        if failed:
            print(f"\nFailed subjects:")
            for r in failed:
                print(f"  - {r['subject_id']}: {r.get('error', 'Unknown error')[:60]}")

            # Save failed list
            failed_file = Path("failed_subjects.txt")
            with open(failed_file, 'w') as f:
                for r in failed:
                    f.write(f"{r['subject_id']}\n")
            print(f"\nFailed subjects saved to: {failed_file}")

        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        logger.warning("Processing interrupted by user")
    except Exception as e:
        print(f"\n\nCritical error: {e}")
        logger.error(f"Critical error: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
