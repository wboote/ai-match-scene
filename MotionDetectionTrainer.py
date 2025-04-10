from flask import Flask, request, jsonify
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime

# ------------------ Configuration ------------------
UPLOAD_FOLDER = 'uploads'
MODEL_SAVE_EVERY = 100
IMAGE_SIZE = (720, 1280)  # 720p resolution: height x width
EPOCHS = 20

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

# ------------------ List Available CUDA Devices ------------------
def list_cuda_devices():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"[INFO] Found {num_devices} CUDA device(s).")
        for i in range(num_devices):
            device_name = torch.cuda.get_device_name(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024 ** 2  # MB
            memory_cached = torch.cuda.memory_reserved(i) / 1024 ** 2  # MB
            print(f"[INFO] Device {i}: {device_name}")
            print(f"  Memory Allocated: {memory_allocated:.2f} MB")
            print(f"  Memory Cached: {memory_cached:.2f} MB")
    else:
        print("[INFO] No CUDA devices found. Running on CPU.")

# ------------------ Init Model & Optimizer ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# List available CUDA devices
list_cuda_devices()

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ------------------ Transform ------------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),  # Resize to 720p
    transforms.ToTensor(),
])

image_counter = 0

# ------------------ Endpoint ------------------
@app.route('/upload', methods=['POST'])
def upload_and_train():
    global image_counter
    try:
        if 'image' not in request.files:
            print("[ERROR] No image file part in the request")
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            print("[ERROR] Empty filename")
            return jsonify({'error': 'Empty filename'}), 400

        image_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{image_counter}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"[INFO] Received image {filename} (#{image_counter})")

        # Load image and train
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        model.train()
        for epoch in range(EPOCHS):
            output = model(image_tensor)
            loss = criterion(output, image_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"[TRAINING] Image #{image_counter}, Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss.item():.6f}")

        # Save model every 100 images
        if image_counter % MODEL_SAVE_EVERY == 0:
            model_path = f"autoencoder_{image_counter}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"[INFO] Saved model to {model_path}")

        return jsonify({'status': 'trained', 'image_id': image_counter}), 200

    except Exception as e:
        print(f"[EXCEPTION] {str(e)}")
        return jsonify({'error': str(e)}), 500

# ------------------ Run Server ------------------
if __name__ == '__main__':
    print("[SERVER] Starting motion detection training server with 720p input...")
    app.run(host='0.0.0.0', port=5000)
