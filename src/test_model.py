import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from custom_model import get_pretrained_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device yang digunakan: {device}")

# Data transformasi dan augmentasi yang sama dengan saat pelatihan
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Memuat dataset pengujian
test_dataset = datasets.ImageFolder('data/train', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Memuat model yang telah dilatih
model = get_pretrained_model(num_classes=4)
model.load_state_dict(torch.load('model_penyakit_tanaman2.pth'))
model.to(device)
model.eval()  # Set model ke mode evaluasi

# Menghitung akurasi pada dataset pengujian
correct = 0
total = 0
with torch.no_grad():  # Tidak perlu menghitung gradient saat evaluasi
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Akurasi pada dataset pengujian: {accuracy * 100:.2f}%')
