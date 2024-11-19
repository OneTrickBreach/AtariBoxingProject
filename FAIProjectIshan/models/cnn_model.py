# models/cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_channels=4, num_actions=18):
        """
        A Convolutional Neural Network for processing Atari Boxing frames.

        Args:
            input_channels (int): Number of stacked input frames (default: 4).
            num_actions (int): Number of possible actions (default: 18 for Boxing).
        """
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Assuming input size (84, 84)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        """
        Forward pass for the CNN.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with action logits.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
