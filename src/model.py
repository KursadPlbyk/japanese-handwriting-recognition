import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    CNN for classifying 28x28 grayscale images of Japanese characters
    (46 hiragana + 49 kanji = 95 classes).

    Architecture: 3 conv+relu+pool blocks (1->32->64->128 channels,
    28x28 -> 14x14 -> 7x7 -> 3x3), then flatten, then 2 fully connected
    layers (1152 -> 256 -> 95).
    """

    def __init__(self):
        super().__init__()

        # Conv layers: padding=1 with kernel_size=3 keeps height/width
        # unchanged, so only the pooling layers shrink the spatial size.
        # Channel count grows (1->32->64->128) as more filters are added.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Shared pooling layer, reused after each conv block.
        # Halves height and width each time (28->14->7->3).
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers.
        # 128 * 3 * 3 = 1152 is the flattened size after the 3rd pool
        # (forced by the architecture, not a free choice).
        self.fc1 = nn.Linear(128 * 3 * 3, 256)

        # Output layer: 95 = number of classes (fixed by the problem,
        # not a free choice).
        self.fc2 = nn.Linear(256, 95)

    def forward(self, x):
        """
        Runs one forward pass: an input image batch through all layers
        to produce raw class scores.

        Args:
            x (Tensor): input images, shape (batch_size, 1, 28, 28)

        Returns:
            Tensor: raw class scores (not probabilities), shape (batch_size, 95)
        """

        x = self.pool(F.relu(self.conv1(x)))  # -> (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch, 64, 7, 7)
        x = self.pool(F.relu(self.conv3(x)))  # -> (batch, 128, 3, 3)
        x = x.view(x.size(0), -1)             # flatten -> (batch, 1152)
        x = F.relu(self.fc1(x))               # -> (batch, 256)
        x = self.fc2(x)                       # -> (batch, 95), no activation here
        return x



    