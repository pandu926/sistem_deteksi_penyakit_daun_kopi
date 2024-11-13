from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
from io import BytesIO
from src.custom_model import get_pretrained_model  # Sesuaikan dengan path model Anda

app = FastAPI()

# Load model A dan model B
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_a = get_pretrained_model(num_classes=2)  # Model untuk deteksi daun
model_b = get_pretrained_model(num_classes=4)  # Model untuk deteksi penyakit

# Load state_dict kedua model
model_a.load_state_dict(torch.load('deteksi_daun.pth', map_location=device))
model_b.load_state_dict(torch.load('model_penyakit_tanaman2.pth', map_location=device))

# Pindahkan kedua model ke perangkat yang sama dan set ke evaluasi
model_a.to(device).eval()
model_b.to(device).eval()

print("Both models loaded successfully.")

# Preprocessing untuk gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Baca gambar dari file yang diupload
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")

    # Terapkan transformasi
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Prediksi dengan model A (deteksi daun)
    with torch.no_grad():
        output_a = model_a(image_tensor)
        _, predicted_class_a = torch.max(output_a, 1)

    if predicted_class_a.item() == 1:  # Asumsikan 1 berarti gambar adalah daun
        # Prediksi dengan model B (deteksi penyakit)
        with torch.no_grad():
            output_b = model_b(image_tensor)
            _, predicted_class_b = torch.max(output_b, 1)

        # Hasil prediksi untuk model B
        class_names_b = ['Cerscospora', 'Healthy', 'Leafrust', 'Phoma']
        predicted_class_name_b = class_names_b[predicted_class_b.item()]

        return JSONResponse(content={"predicted_class": "Daun", "disease_detected": predicted_class_name_b})
    else:
        return JSONResponse(content={"predicted_class": "Bukan Daun", "disease_detected": None})

@app.get("/")
async def root():
    return JSONResponse(content={"message": "API is running."})
