
#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
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

def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
