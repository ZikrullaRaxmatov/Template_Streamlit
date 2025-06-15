from torch import nn

# CNN
class CNN(nn.Module):
    
  def __init__(self):
    super(CNN, self).__init__()
    self.conv_layers = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(64, 128, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(128, 128, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )
    self.fc_layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 9 * 9, 512),
        nn.ReLU(),
        nn.Linear(512, 3),
        nn.Softmax()
    )

  def forward(self, x):
    x = self.conv_layers(x)
    x = self.fc_layers(x)
    return x
