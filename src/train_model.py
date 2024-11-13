import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from custom_model import get_pretrained_model  # Perbaikan import

# Menentukan device (GPU atau CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available())
print(f"Device yang digunakan: {device}")
if device.type == 'cuda':
    print(f"GPU yang digunakan: {torch.cuda.get_device_name(0)}")
    print(f"Memori CUDA yang tersedia sebelum training: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
    print(f"Memori CUDA yang ter-reserved sebelum training: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")

# Data transformasi dan augmentasi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder('data/deteksi_resize', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Inisialisasi model dan parameter training
model = get_pretrained_model(num_classes=2).to(device)  # Memindahkan model ke GPU jika tersedia
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Loop training
model.train()
for epoch in range(10):
    for images, labels in train_loader:
        # Memindahkan data ke GPU jika tersedia
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # Debugging: Memeriksa penggunaan memori GPU sebelum forward pass
        if device.type == 'cuda':
            print(f"Memori CUDA yang digunakan sebelum forward pass: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
            print(f"Memori CUDA yang ter-reserved sebelum forward pass: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Debugging: Memeriksa penggunaan memori GPU setelah backward pass
        if device.type == 'cuda':
            print(f"Memori CUDA yang digunakan setelah backward pass: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
            print(f"Memori CUDA yang ter-reserved setelah backward pass: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # Debugging: Memeriksa penggunaan memori GPU setelah setiap epoch
    if device.type == 'cuda':
        print(f"Memori CUDA yang digunakan setelah epoch {epoch + 1}: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
        print(f"Memori CUDA yang ter-reserved setelah epoch {epoch + 1}: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")

# Menyimpan model yang sudah dilatih
torch.save(model.state_dict(), 'deteksi_daun.pth')
print("Model telah disimpan ke 'deteksi_daun.pth'")
