#!/usr/bin/env python3
"""
Script to merge all hdf5 episodes from processed_data subdirectories 
into a single target directory with renumbered episodes.
"""

import os
import h5py
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def find_all_hdf5_files(source_dir):
    """Find all hdf5 files in source directory and subdirectories."""
    hdf5_files = []
    source_path = Path(source_dir)
    
    for root, dirs, files in os.walk(source_path):
        # Skip the target directory itself to avoid copying files that are already there
        if 'integrated_clean-1200' in root:
            continue
            
        for file in files:
            if file.endswith('.hdf5') and file.startswith('episode_'):
                full_path = os.path.join(root, file)
                hdf5_files.append(full_path)
    
    # Sort files to ensure consistent ordering
    hdf5_files.sort()
    return hdf5_files


def copy_hdf5_file(source_path, target_path):
    """Copy hdf5 file from source to target."""
    shutil.copy2(source_path, target_path)


def merge_episodes(source_dir, target_dir, overwrite=False):
    """Merge all hdf5 episodes from source_dir to target_dir with renumbering."""
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if target directory already has files
    existing_files = [f for f in os.listdir(target_dir) if f.endswith('.hdf5')]
    if existing_files:
        print(f"Warning: Target directory already contains {len(existing_files)} hdf5 files")
        if not overwrite:
            response = input("Do you want to continue and overwrite? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return
        else:
            print("Overwrite mode enabled, proceeding...")
    
    # Find all hdf5 files
    print(f"Scanning for hdf5 files in {source_dir}...")
    hdf5_files = find_all_hdf5_files(source_dir)
    
    if not hdf5_files:
        print(f"No hdf5 files found in {source_dir}")
        return
    
    print(f"Found {len(hdf5_files)} hdf5 files to merge")
    print(f"Target directory: {target_dir}")
    print()
    
    # Copy files with renumbering
    print(f"Copying files to {target_dir}...")
    copied_count = 0
    failed_count = 0
    
    for idx, source_file in enumerate(tqdm(hdf5_files, desc="Copying episodes")):
        # Generate new episode number
        new_episode_num = idx
        target_filename = f"episode_{new_episode_num}.hdf5"
        target_path = os.path.join(target_dir, target_filename)
        
        # Copy file
        try:
            copy_hdf5_file(source_file, target_path)
            copied_count += 1
        except Exception as e:
            print(f"\nError copying {source_file}: {e}")
            failed_count += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Merge completed!")
    print(f"Successfully copied: {copied_count} episodes")
    if failed_count > 0:
        print(f"Failed: {failed_count} episodes")
    print(f"Final episode numbers: 0 to {copied_count - 1}")
    print(f"Target directory: {target_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge all hdf5 episodes from processed_data subdirectories"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/media/liushengbang/ISEE2/policy/ACT/processed_data",
        help="Source directory containing subdirectories with hdf5 files"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/media/liushengbang/ISEE2/policy/ACT/processed_data/sim-six_tasks/integrated_clean-1200",
        help="Target directory to merge all episodes into"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Automatically overwrite existing files without prompting"
    )
    
    args = parser.parse_args()
    
    print(f"Source directory: {args.source_dir}")
    print(f"Target directory: {args.target_dir}")
    print()
    
    merge_episodes(args.source_dir, args.target_dir, args.overwrite)


if __name__ == "__main__":
    main()

