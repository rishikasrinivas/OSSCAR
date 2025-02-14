#!/bin/bash

# Check if folder argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 folder_name"
    exit 1
fi

# Store the folder name from argument
fldr="$1"

# Check if the folder exists
if [ ! -d "$fldr" ]; then
    echo "Error: Directory '$fldr' does not exist"
    exit 1
fi

# Execute the find and copy command
find "$fldr" -type f -not -path '*/\.*' -not -path '*/__*/*' -exec cp --parents {} /tutorial \;

echo "Files copied successfully from '$fldr' to /tutorial"
