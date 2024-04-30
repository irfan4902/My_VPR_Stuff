### Constants ###
WEIGHTS_FILE = "calc.caffemodel.pt"
DATASET = "GardensPoint"
DATABASE_FOLDER = "day_right"
QUERY_FOLDER = "night_right"
ITERATIONS = 50 # for testing average duration
BATCH_SIZE = 32 # dataloader batch_size
NUM_WORKERS = 8 # dataloader num_workers (threads)
DATA_TEXT_FILE = 'data_normal_laptop2.txt' # save the average database and query times to this file


### Imports ###
import os
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import seaborn as sns
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization import quantize_static
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data

from datasets import dataset_utils
from matching import matching
from evaluation.metrics import createPR, recallAt100precision, recallAtK
from datasets.load_dataset import GardensPointDataset, SFUDataset, StLuciaDataset

### Load Dataset ###
# Uncomment to download the desired chosen dataset to the images folder
# GardensPointDataset().load()
# SFUDataset().load()
# StLuciaDataset().load()

# Uncomment this line if using GardensPoint
_, _, gt_hard, gt_soft = GardensPointDataset().load()

# Uncomment these lines if using SFU or StLucia
# gt_data = np.load(f'images/{DATASET}/GT.npz')
# gt_hard = gt_data['GThard']
# gt_soft = gt_data['GTsoft']


### Preprocess Images ###

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

class CustomImageDataset(Dataset):
    def __init__(self, name, folder, transform=None):
        
        self.name = os.path.basename(name)
        self.folder = os.path.join(name, folder)
        self.image_paths = dataset_utils.read_images_paths(self.folder, get_abs_path=True)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index) :
        image_path = self.image_paths[index]
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)
        return(img)

dataset_db = CustomImageDataset(f"images/{DATASET}", DATABASE_FOLDER, preprocess)
dataset_q = CustomImageDataset(f"images/{DATASET}", QUERY_FOLDER, preprocess)

print("Dataset Length:", len(dataset_db))
dataset_db[0]

db_dataloader = DataLoader(dataset_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
q_dataloader = DataLoader(dataset_q, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

### Model Definition ###

class CalcModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = (1, 120, 160)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=2, padding=4)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=1, padding=2)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(128, 4, kernel_size=(3, 3), stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool(x)
        x = self.lrn1(x)

        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.lrn2(x)

        x = self.relu3(self.conv3(x))
        x = torch.flatten(x, 1)
        return x

calc = CalcModel()

# Load the model weights
state_dict = torch.load(WEIGHTS_FILE)
my_new_state_dict = {}
my_layers = list(calc.state_dict().keys())
for layer in my_layers:
    my_new_state_dict[layer] = state_dict[layer]
calc.load_state_dict(my_new_state_dict)

print(calc)


### Run Models ###

def run_model(dataloader, model):
    features = []

    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            features.append(output)

    features = torch.cat(features, axis=0)
    return features

calc.eval()

# Process database tensor
db_features = run_model(db_dataloader, calc)
print(db_features.shape)

# Process query tensor
q_features = run_model(q_dataloader, calc)
print(q_features.shape)


### Average Time ###

def measure_time(dataloader, model, iterations, desc):

    start_time = time.time()

    for _ in tqdm(range(iterations), desc=desc):
        
        run_model(dataloader, model)

    end_time = time.time()

    avg_time = (end_time - start_time) / iterations

    return avg_time

db_time = measure_time(db_dataloader, calc, ITERATIONS, "Processing database dataset")
print(f"Database Average Time: {db_time}")

q_time = measure_time(q_dataloader, calc, ITERATIONS, "Processing query dataset")
print(f"Query Average Time: {q_time}")


### Evaluation ###


# Convert the predictions to a numpy array
features = db_features.detach().cpu().numpy()

# Get the predicted labels (assuming a binary classification problem)
predicted_labels = (features > 0.9).astype(int)

# Get the ground truth labels
true_labels = gt_hard

# Get the predicted labels
predicted_labels = features.argmax(axis=1)

# Calculate the accuracy
accuracy = (predicted_labels == true_labels).mean()
print('Accuracy:', accuracy)

similarity_matrix = cosine_similarity(db_features.detach().numpy(), q_features.detach().numpy())

# best matching per query in S for single-best-match VPR
M1 = matching.best_match_per_query(similarity_matrix)

# find matches with S>=thresh using an auto-tuned threshold for multi-match VPR
M2 = matching.thresholding(similarity_matrix, 0.92)
TP = np.sum(M2 & gt_hard)  # true positives
FP = np.sum(M2 & ~gt_soft)  # false positives
FN = np.sum(gt_hard) - TP

# Calculate precision
precision = TP / (TP + FP)
print('Precision:', precision)

# # Calculate recall
recall = TP / (TP + FN)
print('Recall:', recall)

# Calculate F1 score
f1 = 2 * (precision * recall) / (precision + recall)
print('F1 score:', f1)

# Save data to file
with open(DATA_TEXT_FILE, 'w') as f:
    # Write the database and query times to the file
    f.write(f"Database Average Time: {db_time}\n")
    f.write(f"Query Average Time: {q_time}\n")
