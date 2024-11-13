import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

def get_pretrained_model(num_classes):
    # Memuat model dengan bobot pretrained
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Freeze layers jika tidak ingin melatih ulang feature extractor
    for param in model.parameters():
        param.requires_grad = False
    
    # Ganti layer terakhir sesuai jumlah kelas
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
