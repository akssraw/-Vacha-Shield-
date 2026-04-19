import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for audio deepfake detection.

    UPGRADE 1 (Dual-Channel Input): Now accepts 2-channel feature maps:
      - Channel 0: PCEN Mel-Spectrogram  (energy distribution)
      - Channel 1: Delta Spectrogram     (rate-of-change — TTS smoothness fingerprint)

    The CNN learns to correlate patterns *across both channels simultaneously*,
    giving it the ability to detect high-fidelity clones that fool single-channel models.

    Input shape:  (Batch, 2, 40, 400)
    Output shape: (Batch, 1)  — raw logit; apply sigmoid for probability
    """

    def __init__(self, num_classes=1):
        super(AudioCNN, self).__init__()

        # 1. First Convolutional Block
        # in_channels=2 accepts both the PCEN spectrogram and its delta channel.
        # The conv filters immediately learn cross-channel relationships
        # (e.g., "high energy here + low delta there = TTS artifact").
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after pool1: (16, 20, 200)

        # 2. Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after pool2: (32, 10, 100)

        # 3. Third Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after pool3: (64, 5, 50)

        # 4. Fully Connected Layers
        # Flattened size: 64 * 5 * 50 = 16000
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=64 * 5 * 50, out_features=128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Final output: single logit (apply sigmoid during inference for probability)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor. Shape: (Batch_Size, 2, 40, 400)
               Channel 0 = PCEN spectrogram, Channel 1 = Delta spectrogram
        Returns:
            Raw logit tensor of shape (Batch_Size, 1).
            Apply torch.sigmoid() during inference to get [0, 1] probability.
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)


if __name__ == "__main__":
    # Test: 1 sample, 2 channels (PCEN + Delta), 40 Mel bands, 400 time frames
    dummy_input = torch.randn(1, 2, 40, 400)

    model = AudioCNN()
    output = model(dummy_input)

    print("Model successfully instantiated with dual-channel input!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Raw logit:    {output.item():.4f}")
    print(f"Sigmoid prob: {torch.sigmoid(output).item():.4f}")


