import os
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import onnx
import onnxruntime

from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

from matching import matching
from evaluation.metrics import createPR, recallAt100precision, recallAtK
from datasets.load_dataset import GardensPointDataset, SFUDataset, StLuciaDataset

# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# conda install onnx onnxruntime
# pip install torch torchvision numpy opencv-python seaborn matplotlib scikit-learn pillow onnxscript

imgs_db, imgs_q, GThard, GTsoft = GardensPointDataset().load()
# imgs_db, imgs_q, GThard, GTsoft = SFUDataset().load()
# imgs_db, imgs_q, GThard, GTsoft = StLuciaDataset().load()

print(type(imgs_db))
print(type(imgs_q))
print(type(GThard))
print(type(GTsoft))


preprocess = transforms.Compose([
    # transforms.Resize((360, 640), interpolation=Image.BICUBIC), # not sure if this is good - added it for quantization
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomImageDataset(Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs_db)
    
    def __getitem__(self, idx) :
        img = self.imgs_db[idx]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return(img)

db_tensor = []

for img in imgs_db:
    img = preprocess(Image.fromarray(img))
    db_tensor.append(img)

# Convert list of tensors to a single tensor
db_tensor = torch.stack(db_tensor, axis=0)

print(db_tensor.shape)

# Display the image
plt.imshow(db_tensor[0].permute(1, 2, 0).detach().numpy())
plt.show()

q_tensor = []

for img in imgs_q:
    img = preprocess(Image.fromarray(img))
    q_tensor.append(img)

# Convert list of tensors to a single tensor
q_tensor = torch.stack(q_tensor, axis=0)

print(q_tensor.shape)

# Display the image
plt.imshow(q_tensor[0].permute(1, 2, 0).detach().numpy())
plt.show()

db_dataset = CustomImageDataset(imgs_db, transform=preprocess)
q_dataset = CustomImageDataset(imgs_q, transform=preprocess)

batch_size = 32
db_dataloader = DataLoader(db_tensor, batch_size=batch_size, shuffle=False, num_workers=16)
q_dataloader = DataLoader(q_tensor, batch_size=batch_size, shuffle=False, num_workers=16)

# # Model Definition

model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)

print(model)

print(dir(model))

# set the model to inference mode
model.eval()

example_input = torch.randn(1, 3, 360, 640, requires_grad=True)

example_output = model(example_input)

dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

# Export the model
torch.onnx.export(
    model,  # model
    example_input,  # example input
    "cosplace_model.onnx",  # output file name
    export_params=True,  # export model parameters
    input_names=["input"],  # input names
    output_names=["output"],  # output names
    dynamic_axes=dynamic_axes,  # dynamic axes
)

ort_session = onnxruntime.InferenceSession("cosplace_model.onnx")

# # Run Models

# This cell makes my computer lag

# # Pass database tensor through the model
# db_features = model(db_tensor)
# print(db_features.shape)

# # Pass query tensor through the model
# q_features = model(q_tensor)
# print(q_features.shape)

model.eval()

# Pass database tensor through the model

db_features = []

with torch.no_grad():

    for batch in db_dataloader:
        output = model(batch)
        db_features.append(output)

db_features = torch.cat(db_features, axis=0)

print(db_features.shape)

# Pass query tensor through the model

q_features = []

with torch.no_grad():

    for batch in q_dataloader:
        output = model(batch)
        q_features.append(output)

q_features = torch.cat(q_features, axis=0)

print(q_features.shape)

# # Evaluation

from sklearn.metrics.pairwise import cosine_similarity

db_features_np = db_features.detach().numpy()
q_features_np = q_features.detach().numpy()

# Convert pytorch tensors to numpy arrays
similarity_matrix = cosine_similarity(db_features_np, q_features_np)

print(similarity_matrix.shape)

plt.figure()
sns.heatmap(similarity_matrix, cmap='viridis')
plt.title('Similarity matrix')
plt.axis('off')
plt.show()


