import os
import time

import cv2
import numpy as np
import onnx
import onnxruntime
import quanto
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import torch._compile

from datasets.dataset_utils import read_images_paths
from datasets.load_dataset import GardensPointDataset, SFUDataset, StLuciaDataset
from evaluation.metrics import createPR, recallAt100precision, recallAtK
from matching import matching

WEIGHTS_FILE = "calc.caffemodel.pt"
ITERATIONS = 100 # for testing average duration

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
        self.image_paths = read_images_paths(self.folder, get_abs_path=True)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index) :
        image_path = self.image_paths[index]
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)
        return(img)

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

        @torch.compile
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

if __name__ == '__main__':

    dataset_db = CustomImageDataset("images/SFU", "dry", preprocess)
    dataset_q = CustomImageDataset("images/SFU", "jan", preprocess)

    print("Dataset Length:", len(dataset_db))
    dataset_db[0]

    batch_size = 32
    num_workers = 4
    db_dataloader = DataLoader(dataset_db, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    q_dataloader = DataLoader(dataset_q, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    calc = CalcModel()

    # Load the model weights
    state_dict = torch.load(WEIGHTS_FILE)
    my_new_state_dict = {}
    my_layers = list(calc.state_dict().keys())
    for layer in my_layers:
        my_new_state_dict[layer] = state_dict[layer]
    calc.load_state_dict(my_new_state_dict)

    print(calc)

    # Compile the model
    calc_compiled = torch.compile(calc)

    # Run Models

    calc.eval()

    # Pass database tensor through the model

    db_features = []

    with torch.no_grad():

        for batch in db_dataloader:
            output = calc(batch)
            db_features.append(output)

    db_features = torch.cat(db_features, axis=0)

    print(db_features.shape)

    # Pass query tensor through the model

    q_features = []

    with torch.no_grad():

        for batch in q_dataloader:
            output = calc(batch)
            q_features.append(output)

    q_features = torch.cat(q_features, axis=0)

    print(q_features.shape)

    # Average Time

    # Normal Model

    times = [] # Initialize a list to store the time for each pass

    for _ in tqdm(range(ITERATIONS), desc="Processing database dataset"):
        start_time = time.time()

        with torch.no_grad():
            db_features = []
            for batch in db_dataloader:
                output = calc(batch)
                db_features.append(output)

        end_time = time.time()
        times.append(end_time - start_time)

    average_time = sum(times) / len(times)

    print(f'Average time: {average_time} seconds')

    times = [] # Initialize a list to store the time for each pass

    for _ in tqdm(range(ITERATIONS), desc="Processing query dataset"):
        start_time = time.time()

        with torch.no_grad():
            for batch in q_dataloader:
                output = calc(batch)

        end_time = time.time()
        times.append(end_time - start_time)

    average_time = sum(times) / len(times)

    print(f'Average time: {average_time} seconds')