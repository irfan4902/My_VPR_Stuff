import torch

# model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)

print(model)

print(torch.hub)