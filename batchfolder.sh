#!/bin/bash

# Check if the folder path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

# Assign the folder path to a variable
folder_path="$1"

# Process each .wav file in the folder using full paths
for file in "$folder_path"/*.wav; do
    # Check if files exist (prevent processing literal "*.wav" if no matches)
    if [ -f "$file" ]; then
        python inference.py --audio_path "$file"
    fi
done