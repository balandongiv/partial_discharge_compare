"""
Station Discovery Utility for Partial Discharge Classification Pipeline

This utility module provides automated discovery of available measurement stations
in the dataset. It scans the dataset directory structure to identify all stations
containing partial discharge measurement data, enabling dynamic processing of
available data without hardcoded station lists.

Step-by-Step Process:
1. Directory Structure Validation:
   - Checks for existence of dataset/contactless_pd_detection/ base directory
   - Validates that the expected dataset structure is present
   - Returns empty list if base directory doesn't exist

2. Station Directory Scanning:
   - Iterates through all subdirectories in contactless_pd_detection/
   - Identifies directories with "station_" prefix naming convention
   - Filters for actual directories (excludes files and other entries)

3. Station ID Extraction:
   - Parses station directory names to extract numeric IDs
   - Removes "station_" prefix to get clean station identifiers
   - Handles various station ID formats and naming patterns

4. Station ID Sorting:
   - Sorts discovered station IDs in ascending order
   - Ensures consistent processing order across pipeline runs
   - Provides predictable station processing sequence

5. Result Compilation:
   - Returns list of station ID strings
   - Enables dynamic station processing in pipeline stages
   - Supports both single-station and multi-station processing modes

Usage Scenarios:
- Automatic discovery of all available stations for full pipeline runs
- Validation of specific station existence before processing
- Dynamic dataset exploration and validation
- Support for expanding datasets with new stations

Directory Structure Expected:
```
dataset/
└── contactless_pd_detection/
    ├── station_52008/
    ├── station_52009/
    ├── station_52010/
    └── ...
```

Configuration Parameters:
- dataset_root: Path to dataset root directory
- base_path: Relative path to contactless_pd_detection directory

Dependencies:
- pathlib: Cross-platform path handling and directory operations

Return Value:
- List of station ID strings (e.g., ['52008', '52009', '52010'])
- Empty list if no stations found or directory doesn't exist
- Sorted in ascending order for consistent processing

Benefits:
- Dynamic dataset discovery without hardcoded station lists
- Robust handling of missing or malformed directory structures
- Consistent station ID extraction and sorting
- Easy integration with pipeline processing loops
"""

from __future__ import annotations

from pathlib import Path


def discover_station_ids(dataset_root: Path) -> list[str]:
    ids: list[str] = []
    base = dataset_root / "contactless_pd_detection"
    if not base.exists():
        return ids
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("station_"):
            ids.append(p.name.split("station_")[-1])
    return sorted(ids)


