#!/bin/bash
# Config
SERVER_URL="http://192.168.1.2:5000"
NUM_IMAGES=2000
SAVE_DIR="$HOME/timelapse_photos"
PROGRESS_BAR_WIDTH=50
UPLOAD_TIMEOUT=30  # Timeout for uploads in seconds
CAMERA_PORT="usb:005,010"  # Using the specific port from the first script

# Create save directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Check for gphoto2
if ! command -v gphoto2 &> /dev/null; then
    echo "gphoto2 not found. Please install it first."
    exit 1
fi

# Check for curl
if ! command -v curl &> /dev/null; then
    echo "curl not found. Please install it first."
    exit 1
fi

# Test camera connection
echo "[INFO] Testing camera connection on port $CAMERA_PORT..."
if ! gphoto2 --port "$CAMERA_PORT" --summary &> /dev/null; then
    echo "[ERROR] Camera connection test failed on port $CAMERA_PORT."
    exit 1
fi

# Check server connection
echo "[INFO] Testing server connection..."
if ! curl -s --connect-timeout 5 "$SERVER_URL" &> /dev/null; then
    echo "[ERROR] Cannot connect to server at $SERVER_URL"
    exit 1
fi

# Initialize counters
success_count=0
error_count=0

# Progress bar function
update_progress() {
    local progress=$1
    local filled=$((progress * PROGRESS_BAR_WIDTH / 100))
    local empty=$((PROGRESS_BAR_WIDTH - filled))
    
    # Build progress bar
    progress_bar="["
    for ((i=0; i<filled; i++)); do
        progress_bar+="#"
    done
    for ((i=0; i<empty; i++)); do
        progress_bar+="-"
    done
    progress_bar+="]"
    
    printf "\rProgress: %s %d%% (Success: %d, Errors: %d)" "$progress_bar" "$progress" "$success_count" "$error_count"
}

# Main capture and upload loop
echo "[INFO] Starting capture and upload process using camera at port $CAMERA_PORT..."
for ((i=1; i<=NUM_IMAGES; i++)); do
    echo ""
    # Create timestamp for unique filename
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    CURRENT_IMAGE="$SAVE_DIR/photo_$TIMESTAMP.jpg"
    
    echo "[INFO] Capturing image $i of $NUM_IMAGES..."
    
    # Using the capture method from the first script - with specific port
    if ! gphoto2 --port "$CAMERA_PORT" --capture-image-and-download --filename "$CURRENT_IMAGE" &> /dev/null; then
        echo "[ERROR] Failed to capture image."
        error_count=$((error_count + 1))
        continue
    fi
    
    # Verify file exists
    if [[ ! -f "$CURRENT_IMAGE" ]]; then
        echo "[ERROR] Image file not found after capture."
        error_count=$((error_count + 1))
        continue
    fi
    
    echo "[INFO] Uploading to server..."
    # Attempt upload with timeout
    upload_result=$(curl -s -m "$UPLOAD_TIMEOUT" -F "image=@$CURRENT_IMAGE" "$SERVER_URL/upload")
    
    if [[ $? -ne 0 || -z "$upload_result" ]]; then
        echo "[ERROR] Upload failed or timed out."
        error_count=$((error_count + 1))
        continue
    fi
    
    # Upload successful
    success_count=$((success_count + 1))
    
    # Calculate and show local progress
    current_progress=$((i * 100 / NUM_IMAGES))
    update_progress "$current_progress"
    
    # Optional: add a small delay between captures
    sleep 1
done

echo -e "\n[INFO] Process completed. Total: $NUM_IMAGES, Success: $success_count, Errors: $error_count"
