#!/bin/bash 

SERVER_URL="http://192.168.1.2:5000/predict"
FILENAME="capture.jpg"

while true; do
    echo "[INFO] Capturing image..."
    
    # Capture and download image from camera
    gphoto2 --capture-image-and-download --filename "$FILENAME" --force-overwrite

    if [[ -f "$FILENAME" ]]; then
        echo "[INFO] Sending image to prediction endpoint..."
        
        # Send the image to the prediction endpoint
        RESPONSE=$(curl -s -X POST -F "image=@$FILENAME" "$SERVER_URL")
        echo "[PREDICT RESPONSE] $RESPONSE"

        # Delete image from camera
        gphoto2 --delete-all-files --recurse || echo "[WARN] Couldn't delete from camera"

        # Delete local image
        rm "$FILENAME"
    else
        echo "[ERROR] Failed to capture image."
    fi

    sleep 7
done
