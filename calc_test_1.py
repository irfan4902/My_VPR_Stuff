from datasets.load_dataset import GardensPointDataset, SFUDataset

import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

WEIGHTS_FILE = "calc.caffemodel.pt"

################################## Pre-process #######################

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

################################# Model Definition ############################

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

def main():

    # Instantiate the Model
    calc = CalcModel()
    
    # Load the model weights
    state_dict = torch.load(WEIGHTS_FILE)
    my_new_state_dict = {}
    my_layers = list(calc.state_dict().keys())
    for layer in my_layers:
        my_new_state_dict[layer] = state_dict[layer]
    calc.load_state_dict(my_new_state_dict)

    SFUDataset().load()

    # Preprocess query images

    query_folder = os.listdir('images\SFU\jan')
    query_matrix = []

    for image in query_folder:
        image_path = os.path.join('images\SFU\jan', image)
        image = Image.open(image_path)
        preprocessed_image = preprocess(image)
        query_matrix.append(preprocessed_image)

    query_matrix = np.stack(query_matrix, axis=0)

    print(query_matrix.shape)
    print(query_matrix)


    # Preprocess map images

    map_folder = os.listdir('images\SFU\dry')
    map_matrix = []

    for image in map_folder:
        image_path = os.path.join('images\SFU\dry', image)
        image = Image.open(image_path)
        preprocessed_image = preprocess(image)
        map_matrix.append(preprocessed_image)

    map_matrix = np.stack(map_matrix, axis=0)

    print(map_matrix.shape)
    print(map_matrix)

if __name__ == "__main__":
    main()
