#!/bin/bash

# Merge all zarr datasets into a single integrated dataset
cd "$(dirname "$0")"

python merge_zarr_data.py

