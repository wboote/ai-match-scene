from flask import Flask, request, jsonify
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime

# ------------------ Configuration ------------------
UPLOAD_FOLDER = 'uploads'
IMAGE_SIZE = (720, 1280)  # 720p resolution: height x width
EPOCHS = 8
EARLY_STOP_LOSS_THRESHOLD = 0.0034  # very low loss expected for normal scenes
MAX_IMAGES_WITHOUT_SAVING = 400
MODEL_PATH = "autoencoder_final.pt"

# ------------------ Flask App ------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------ U-Net Style Autoencoder for Anomaly Detection ------------------
# This architecture has skip connections that help in preserving spatial detail.
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # 720x1280 -> 360x640

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)  # 360x640 -> 180x320

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)  # 180x320 -> 90x160

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)      # [B, 64, 720, 1280]
        p1 = self.pool1(x1)         # [B, 64, 360, 640]

        x2 = self.enc_conv2(p1)     # [B, 128, 360, 640]
        p2 = self.pool2(x2)         # [B, 128, 180, 320]

        x3 = self.enc_conv3(p2)     # [B, 256, 180, 320]
        p3 = self.pool3(x3)         # [B, 256, 90, 160]

        bottleneck = self.bottleneck(p3)  # [B, 512, 90, 160]

        # Decoder with skip connections
        d3 = self.up3(bottleneck)         # [B, 256, 180, 320]
        d3 = torch.cat([d3, x3], dim=1)     # [B, 512, 180, 320]
        d3 = self.dec_conv3(d3)            # [B, 256, 180, 320]

        d2 = self.up2(d3)                 # [B, 128, 360, 640]
        d2 = torch.cat([d2, x2], dim=1)     # [B, 256, 360, 640]
        d2 = self.dec_conv2(d2)            # [B, 128, 360, 640]

        d1 = self.up1(d2)                 # [B, 64, 720, 1280]
        d1 = torch.cat([d1, x1], dim=1)     # [B, 128, 720, 1280]
        d1 = self.dec_conv1(d1)            # [B, 64, 720, 1280]

        # Final output layer
        out = self.out_conv(d1)            # [B, 3, 720, 1280]
        return out

# ------------------ Init Model & Optimizer ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

model = UNetAutoencoder().to(device)

# Load model weights if available
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[INFO] Loaded model weights from {MODEL_PATH}")
else:
    print(f"[INFO] No pretrained model found at {MODEL_PATH}, starting fresh.")

criterion = nn.L1Loss()  # L1 loss helps emphasize small differences
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ------------------ Transform ------------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# ------------------ Counters ------------------
image_counter = 0
images_since_last_save = 0

# ------------------ Training Endpoint ------------------
@app.route('/upload', methods=['POST'])
def upload_and_train():
    global image_counter, images_since_last_save
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        image_counter += 1
        images_since_last_save += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{image_counter}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"[INFO] Received image {filename} (#{image_counter})")

        # Preprocess image
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        model.train()
        saved_early = False
        for epoch in range(EPOCHS):
            output = model(image_tensor)
            loss = criterion(output, image_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[TRAINING] Image #{image_counter}, Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss.item():.12f}")

            # Early stopping if the loss is already sufficiently low
            if loss.item() < EARLY_STOP_LOSS_THRESHOLD:
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"[INFO] Early stopping: loss={loss.item():.12f}. Saved model to {MODEL_PATH}")
                images_since_last_save = 0
                saved_early = True
                break

        # Save model every MAX_IMAGES_WITHOUT_SAVING new images
        if images_since_last_save >= MAX_IMAGES_WITHOUT_SAVING:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[INFO] Saved model to {MODEL_PATH} after {images_since_last_save} new images.")
            images_since_last_save = 0

        # Calculate a confidence score for anomaly detection:
        # For anomaly detection, a well-reconstructed (normal) image should result in low loss.
        # Here we invert and normalize the loss (adjust MAX_EXPECTED_LOSS as needed).
        MAX_EXPECTED_LOSS = 0.01  # Adjust based on your training observations
        confidence = max(0.0, 1.0 - (loss.item() / MAX_EXPECTED_LOSS))
        
        return jsonify({
            'status': 'trained',
            'loss': round(loss.item(), 12),
            'confidence': confidence,
            'image_id': image_counter
        }), 200

    except Exception as e:
        print(f"[EXCEPTION] {str(e)}")
        return jsonify({'error': str(e)}), 500

# ------------------ Run Server ------------------
if __name__ == '__main__':
    print("[SERVER] Starting anomaly detection training server with 720p input...")
    app.run(host='0.0.0.0', port=5000)
