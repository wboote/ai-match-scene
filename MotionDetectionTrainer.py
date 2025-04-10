from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import os

# Settings
IMAGE_SIZE = (256, 256)
NUM_EPOCHS = 20
TOTAL_IMAGES = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   # -> [16, 128, 128]
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> [32, 64, 64]
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # -> [16, 128, 128]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),   # -> [3, 256, 256]
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate model, loss, optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Transform image
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# Progress state
received_images = 0
pbar = tqdm(total=TOTAL_IMAGES, desc="Training", unit="image")

@app.route('/upload', methods=['POST'])
def upload_image():
    global received_images

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Train on the single image for NUM_EPOCHS
    model.train()
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        output = model(image_tensor)
        loss = criterion(output, image_tensor)
        loss.backward()
        optimizer.step()

    received_images += 1
    pbar.update(1)

    # Stop condition
    if received_images >= TOTAL_IMAGES:
        pbar.close()
        print("[INFO] Finished training on all images.")
        os._exit(0)

    return jsonify({'status': 'success', 'trained': received_images})

@app.route('/progress', methods=['GET'])
def get_progress():
    percent = int((received_images / TOTAL_IMAGES) * 100)
    return jsonify({'progress': percent})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
