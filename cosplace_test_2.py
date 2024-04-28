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

BATCH_SIZE=32
NUM_WORKERS=4

preprocess = transforms.Compose([
    # transforms.Resize((360, 640), interpolation=Image.BICUBIC), # not sure if this is good - added it for quantization
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

if __name__ == '__main__':

    dataset_db = CustomImageDataset("images/SFU", "dry", preprocess)
    dataset_q = CustomImageDataset("images/SFU", "jan", preprocess)

    offset = dataset_db.__len__() // 2
    calib_ds = torch.utils.data.Subset(dataset_db, list(range(offset)))
    val_ds = torch.utils.data.Subset(dataset_db, list(range(offset, offset * 2)))

    print("Dataset Length:", len(dataset_db))
    image = dataset_db[0]
    print("Image Shape:", image.shape)

    db_dataloader = DataLoader(dataset_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    q_dataloader = DataLoader(dataset_q, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    #
    # Load Model
    #
    model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
    model.eval()
    print(model)

    #
    # Quantize Model
    #
    quantized_model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
    quanto.quantize(quantized_model, weights=quanto.qint8, activations=quanto.qint8)
    quantized_model.eval()
    print(quantized_model)
    quanto.freeze(quantized_model)

    #
    # Run Model
    #
    start_time = time.time()

    db_features = []

    with torch.no_grad():

        for batch in tqdm(db_dataloader):
            output = model(batch)
            db_features.append(output)

    db_features = torch.cat(db_features, axis=0)

    end_time = time.time()

    print(db_features.shape)
    print("Duration:", end_time - start_time)

    #
    # Run Quantized Model
    #
    start_time = time.time()

    db_features = []

    with torch.no_grad():

        for batch in tqdm(db_dataloader):
            output = quantized_model(batch)
            db_features.append(output)

    db_features = torch.cat(db_features, axis=0)

    end_time = time.time()

    print(db_features.shape)
    print("Duration:", end_time - start_time)
