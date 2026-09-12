import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()

        # Convolutional part
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75,k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv 3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv 4
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),

            # Conv 5
            nn.Conv2d(384, 256, kernel_size=3, padding=1,groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Fully connected part
        self.classifier = nn.Sequential(

            # FC 1
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            # FC 2
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # FC 3
            nn.Linear(4096, num_classes)
        )

        self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        # Paper initialization:
        # weights ~ N(0, 0.01)
        # biases are initially 0
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(
                    layer.weight,
                    mean=0.0,
                    std=0.01
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
        # The paper initializes these biases to 1
        nn.init.constant_(self.features[4].bias, 1.0)   # Conv2
        nn.init.constant_(self.features[10].bias, 1.0)  # Conv4
        nn.init.constant_(self.features[12].bias, 1.0)  # Conv5
        nn.init.constant_(self.classifier[1].bias, 1.0) # FC1
        nn.init.constant_(self.classifier[4].bias, 1.0) # FC2

# Create model
model = AlexNet(num_classes=1000)

print(model)