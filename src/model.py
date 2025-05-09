import torch.nn as nn
import torchvision.models as models

class PopcornLungCNN(nn.Module):
    def __init__(self):
        super(PopcornLungCNN, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)

    def forward(self, x):
        return self.base_model(x)
