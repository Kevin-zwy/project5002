# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image

class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        # load ResNet50
        self.base_model = models.resnet50(weights='IMAGENET1K_V1')


        self.base_model.fc = nn.Identity()


        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.base_model(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomDenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet121, self).__init__()
        # load DenseNet121
        self.base_model = models.densenet121(weights='IMAGENET1K_V1')

        # Remove the classification layer
        self.base_model.classifier = nn.Identity()

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.base_model(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(CustomMobileNetV2, self).__init__()

        # load MobileNetV2
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')

        self.base_model.classifier = nn.Identity()

        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.base_model(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def predict_image_voting(image_path, models, device, class_names):

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)

    predictions = []

    for model in models:
        model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())

    final_prediction = max(set(predictions), key=predictions.count)

    predicted_class_name = class_names[final_prediction]

    return predicted_class_name
