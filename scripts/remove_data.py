#!/usr/bin/env python3
"""
Project cleanup script to remove files from data, models, and plots directories.
By default, removes files from all three directories.
Use --plots, --data, --models to specify which directories to clean.
"""

import argparse
import os
import shutil

def remove_files_in_directory(directory):
    """Remove all files in the given directory recursively."""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Clean up project directories.")
    parser.add_argument('--plots', action='store_true', help="Remove files in plots directory")
    parser.add_argument('--data', action='store_true', help="Remove files in data directory")
    parser.add_argument('--models', action='store_true', help="Remove files in src/models directory")

    args = parser.parse_args()

    # If no args specified, clean all
    clean_all = not (args.plots or args.data or args.models)

    if clean_all or args.plots:
        remove_files_in_directory('plots')
    if clean_all or args.data:
        remove_files_in_directory('data')
    if clean_all or args.models:
        remove_files_in_directory('src/models')

if __name__ == "__main__":
    main()