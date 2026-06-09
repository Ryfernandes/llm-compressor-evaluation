"""
Directory Scanner for Evaluation Results

Recursively discovers JSON files in directory trees.
"""

import os
from pathlib import Path


def scan_json_files(directories: list[str]) -> list[str]:
    """
    Given a list of directory paths (relative or absolute), returns all absolute paths of JSON
    files in those directories (and subdirectories)
    """
    json_files = []

    for directory in directories:
        dir_path = Path(directory).resolve()

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Recursively find all .json files
        for json_file in dir_path.rglob("*.json"):
            if json_file.is_file():
                json_files.append(str(json_file))

    return sorted(json_files)
