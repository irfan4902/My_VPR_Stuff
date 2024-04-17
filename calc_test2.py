# %% [markdown]
# # Import Stuff

# %%
import os
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
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
from datasets.load_dataset import GardensPointDataset, SFUDataset, StLuciaDataset
from evaluation.metrics import createPR, recallAt100precision, recallAtK
from evaluation import show_correct_and_wrong_matches

# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# conda install onnx onnxruntime
# pip install torch torchvision numpy opencv-python seaborn matplotlib scikit-learn scikit-image pillow onnxscript

# %% [markdown]
# # Load Datasets

# %%
imgs_db, imgs_q, GThard, GTsoft = SFUDataset().load()
# imgs_db, imgs_q, GThard, GTsoft = GardensPointDataset().load()
# imgs_db, imgs_q, GThard, GTsoft = StLuciaDataset().load()
print("Loaded datasets")

# %% [markdown]
# # Constants

# %%
WEIGHTS_FILE = "calc.caffemodel.pt"
ITERATIONS = 100

# %% [markdown]
# # Model Definition

# %%
class CalcModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = (1, 120, 160)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=2, padding=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=1, padding=2)
        self.conv3 = nn.Conv2d(128, 4, kernel_size=(3, 3), stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.lrn1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.lrn2(x)

        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        return x

# %% [markdown]
# ### Normal Model

# %%
calc = CalcModel()

# Load the model weights
state_dict = torch.load(WEIGHTS_FILE)
my_new_state_dict = {}
my_layers = list(calc.state_dict().keys())
for layer in my_layers:
    my_new_state_dict[layer] = state_dict[layer]
calc.load_state_dict(my_new_state_dict)

# %% [markdown]
# # Preprocess Images

# %%
class ConvertToYUVandEqualizeHist:
    def __call__(self, img):
        img_yuv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return Image.fromarray(img_rgb)

preprocess = transforms.Compose(
    [
        ConvertToYUVandEqualizeHist(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((120, 160), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ]
)

# %%
# Preprocess query images

query_matrix = []

for image in imgs_db:
    preprocessed_image = preprocess(image)
    query_matrix.append(preprocessed_image)

query_matrix = np.stack(query_matrix, axis=0)

# Convert numpy array to a tensor
query_tensor = torch.from_numpy(query_matrix)

print(query_tensor.shape)
print(query_tensor)

# %%
# Preprocess map images

map_matrix = []

for image in imgs_q:
    preprocessed_image = preprocess(image)
    map_matrix.append(preprocessed_image)

map_matrix = np.stack(map_matrix, axis=0)

# Convert numpy array to a tensor
map_tensor = torch.from_numpy(map_matrix)

print(map_tensor.shape)
print(map_tensor)

# %% [markdown]
# # Run Models

# %% [markdown]
# ### Normal Model

# %%
query_tensor = torch.from_numpy(query_matrix)
map_tensor = torch.from_numpy(map_matrix)

print(query_tensor.shape)
print(map_tensor.shape)

query_tensor = query_tensor.view(-1, 1, 120, 160)
map_tensor = map_tensor.view(-1, 1, 120, 160)

# Pass the tensors through the model

query_features = calc(query_tensor)
map_features = calc(map_tensor)

print(query_features.shape)
print(map_features.shape)

# %%
# Convert pytorch tensors to numpy arrays
query_features_np = query_features.detach().numpy()
map_features_np = map_features.detach().numpy()

similarity_matrix = cosine_similarity(query_features_np, map_features_np)

print(similarity_matrix.shape)

# %%
plt.figure()
sns.heatmap(similarity_matrix, cmap='viridis')
plt.title('Similarity matrix')
plt.axis('off')
plt.show()

# %%
# best matching per query in S for single-best-match VPR
M1 = matching.best_match_per_query(similarity_matrix)

# find matches with S>=thresh using an auto-tuned threshold for multi-match VPR
M2 = matching.thresholding(similarity_matrix, 'auto')

TP = np.argwhere(M2 & GThard)  # true positives
FP = np.argwhere(M2 & ~GTsoft)  # false positives

# show M's
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(M1)
ax1.axis('off')
ax1.set_title('Best match per query')
ax2 = fig.add_subplot(122)
ax2.imshow(M2)
ax2.axis('off')
ax2.set_title('Thresholding S>=thresh')

# precision-recall curve
P, R = createPR(similarity_matrix, GThard, GTsoft, matching='multi', n_thresh=100)
plt.figure()
plt.plot(R, P)
plt.xlim(0, 1), plt.ylim(0, 1.01)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Result on GardensPoint day_right--night_right')
plt.grid('on')
plt.draw()
plt.show()