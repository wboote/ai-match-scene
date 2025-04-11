from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import os

# ------------------ Configuration ------------------
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'autoencoder_final.pt'
IMAGE_SIZE = (720, 1280)  # 720p resolution: height x width
EPOCHS = 20
CONFIDENCE_THRESHOLD = 0.99755  # Confidence threshold to determine match

# ------------------ Flask App ------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------ Autoencoder Model ------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ------------------ Init Model & Optimizer ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

model = Autoencoder().to(device)

# Load the pre-trained model at startup
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[INFO] Loaded model from {MODEL_PATH}")
else:
    print("[ERROR] Model file not found!")

# ------------------ Loss Function & Optimizer ------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ------------------ Transform ------------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # Resize to 720p
    transforms.ToTensor(),
])

# ------------------ Predict Endpoint ------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        image = Image.open(file.stream).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            loss = torch.nn.functional.mse_loss(output, image_tensor)

        # Compute confidence as 1 - loss
        confidence = 1 - loss.item()

        # Format confidence to 12 decimal points
        confidence_rounded = round(confidence, 12)

        # Classification interpretation based on the confidence threshold
        if confidence_rounded > CONFIDENCE_THRESHOLD:
            classification = "Match"
        else:
            classification = "Not a Match"

        return jsonify({
            'status': 'success',
            'confidence': f'{confidence_rounded:.12f}',  # 12 decimal places
            'classification': classification
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------ Run Server ------------------
if __name__ == '__main__':
    print("[SERVER] Starting motion detection training server with 720p input...")
    app.run(host='0.0.0.0', port=5000)
