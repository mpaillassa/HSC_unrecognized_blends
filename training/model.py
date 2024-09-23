import torch
from torch import nn


class BlendIdentifier(nn.Module):
    """Blend identifier model class."""

    def __init__(self, drop_rate):
        """Initializes the blend identifier model class.

        Args:
            - drop_rate (float): rate for dropout in dense layers.
        """
        super().__init__()

        # conv layers
        self.conv1 = nn.Conv2d(5, 32, (3, 3))
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 128, (3, 3))
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.relu4 = nn.ReLU()

        # dense layers
        self.linear1 = nn.Linear(2048, 512)  # after flattening 4*4*128 = 2048
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=drop_rate)

        self.linear2 = nn.Linear(512, 512)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=drop_rate)

        self.linear3 = nn.Linear(512, 2)
        self.softmax = nn.Softmax(1)

    def forward(self, inputs):
        """Makes a forward pass with the model.

        Args:
            inputs (torch.Tensor): input images.
        Returns:
            out (torch.Tensor): prediction.
        """

        c1 = self.conv1(inputs)
        p1 = self.pool1(c1)
        a1 = self.relu1(p1)

        c2 = self.conv2(a1)
        p2 = self.pool2(c2)
        a2 = self.relu2(p2)

        c3 = self.conv3(a2)
        p3 = self.pool3(c3)
        a3 = self.relu3(p3)

        c4 = self.conv4(a3)
        p4 = self.pool4(c4)
        a4 = self.relu4(p4)

        f = torch.flatten(a4, 1, 3)

        l1 = self.linear1(f)
        la1 = self.act1(l1)
        d1 = self.drop1(la1)

        l2 = self.linear2(d1)
        la2 = self.act2(l2)
        d2 = self.drop2(la2)

        l3 = self.linear3(d2)
        out = self.softmax(l3)

        return out
