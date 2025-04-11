#!/bin/bash

# Bash script to send the first .jpg file from the uploads directory to the /predict endpoint

# Configuration
UPLOAD_DIR="uploads"              # Directory containing .jpg files
URI="http://192.168.1.2:5000/predict"  # Predict endpoint URL

# Check if the uploads directory exists
if [ ! -d "$UPLOAD_DIR" ]; then
    echo "Error: Directory does not exist: $UPLOAD_DIR" >&2
    exit 1
fi

# Find the first .jpg file in the uploads directory
file=$(find "$UPLOAD_DIR" -maxdepth 1 -type f -name "*.jpg" | head -n 1)

# Check if a .jpg file was found
if [ -z "$file" ]; then
    echo "Error: No .jpg files found in $UPLOAD_DIR" >&2
    exit 1
fi

filename=$(basename "$file")
echo "Processing $filename"

# Send POST request with curl to the /predict endpoint
echo "Sending $filename to $URI..."
response=$(curl -s -X POST \
    -F "image=@$file" \
    "$URI" 2>/dev/null)

# Check if curl succeeded
if [ $? -ne 0 ]; then
    echo "Error: Failed to send $filename: Network or server error" >&2
    exit 1
fi

# Parse JSON response using jq if available
if command -v jq >/dev/null 2>&1; then
    status=$(echo "$response" | jq -r '.status // "unknown"')
    loss=$(echo "$response" | jq -r '.loss // "unknown"')
    confidence=$(echo "$response" | jq -r '.confidence // "unknown"')
    echo "Response for $filename: Status=$status, Loss=$loss, Confidence=$confidence"
else
    echo "Response for $filename: Raw response: $response"
    echo "Warning: Install jq for better JSON parsing" >&2
fi