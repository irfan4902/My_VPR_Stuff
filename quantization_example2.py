import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

# Define a simple convolutional neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Train the model
def train(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Test the model
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# Create and train the non-quantized model
non_quantized_model = ConvNet()
print(non_quantized_model)

optimizer = torch.optim.Adam(non_quantized_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

start_time = time.time()
train(non_quantized_model, train_loader, optimizer, criterion, epochs=5)
non_quantized_model_time = time.time() - start_time

test_loss, accuracy = test(non_quantized_model, test_loader)
print('Non-quantized model:')
print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, accuracy))
print('Time taken for training and evaluation: {:.2f} seconds'.format(non_quantized_model_time))

# Create and train the quantized model (post-training dynamic weight-only quantization)
quantized_model = torch.quantization.quantize_dynamic(
    non_quantized_model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
)
print(quantized_model)

start_time = time.time()
test_loss, accuracy = test(quantized_model, test_loader)
quantized_model_time = time.time() - start_time

print('\nQuantized model (post-training dynamic weight-only quantization):')
print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, accuracy))
print('Time taken for inference: {:.2f} seconds'.format(quantized_model_time))
