from flask import Flask, request, jsonify
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime

# ------------------ Configuration ------------------
UPLOAD_FOLDER = 'uploads'
IMAGE_SIZE = (720, 1280)
EPOCHS = 8
EARLY_STOP_LOSS_THRESHOLD = 0.001
MAX_IMAGES_WITHOUT_SAVING = 1000

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
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ------------------ Transform ------------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

image_counter = 0

# ------------------ Endpoint ------------------
@app.route('/upload', methods=['POST'])
def upload_and_train():
    global image_counter
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        image_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{image_counter}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"[INFO] Received image {filename} (#{image_counter})")

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

            if loss.item() < EARLY_STOP_LOSS_THRESHOLD:
                torch.save(model.state_dict(), "autoencoder_final.pt")
                print(f"[INFO] Early stopping: loss={loss.item():.12f}. Saved model to autoencoder_final.pt")
                saved_early = True
                break

        # Fallback: save after 1000 images if loss hasn't hit threshold
        if image_counter >= MAX_IMAGES_WITHOUT_SAVING and not saved_early:
            torch.save(model.state_dict(), "autoencoder_final.pt")
            print(f"[INFO] Reached {MAX_IMAGES_WITHOUT_SAVING} images. Saved model to autoencoder_final.pt")

        return jsonify({
            'status': 'trained',
            'loss': round(loss.item(), 12),
            'image_id': image_counter
        }), 200

    except Exception as e:
        print(f"[EXCEPTION] {str(e)}")
        return jsonify({'error': str(e)}), 500

# ------------------ Run Server ------------------
if __name__ == '__main__':
    print("[SERVER] Starting motion detection training server with 720p input...")
    app.run(host='0.0.0.0', port=5000)
