### Constants ###
WEIGHTS_FILE = "calc.caffemodel.pt"
DATASET = "GardensPoint"
DATABASE_FOLDER = "day_right"
QUERY_FOLDER = "night_right"
ITERATIONS = 100 # for testing average duration
BATCH_SIZE = 64 # dataloader batch_size
NUM_WORKERS = 8 # dataloader num_workers (threads)
DATA_TEXT_FILE = 'data_quant_static_desktop.txt' # save the average database and query times to this file


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

example_input = torch.randn(1, 1, 120, 160)

dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

# Export the model
torch.onnx.export(
    calc,  # model
    example_input,  # example input
    "calc_model.onnx",  # output file name
    input_names=["input"],  # input names
    output_names=["output"],  # output names
    dynamic_axes=dynamic_axes,  # dynamic axes
)

ort_session = onnxruntime.InferenceSession("calc_model.onnx")

quant_pre_process('calc_model.onnx', 'calc_model_quant_static_prep.onnx')

calib_ds = torch.stack([dataset_db[i] for i in range(100)])
val_ds = torch.stack([dataset_db[i] for i in range(100, len(dataset_db))])

print(calib_ds.shape)
print(val_ds.shape)

class QuantizationDataReader(CalibrationDataReader):
    def __init__(self, torch_ds, batch_size, input_name):
        self.torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, shuffle=False)
        self.input_name = input_name
        self.datasize = len(self.torch_dl)
        self.enum_data = iter(self.torch_dl)

    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:

            data = self.to_numpy(batch[0])
            data = np.expand_dims(data, axis=0)  # Add a new dimension to the data
            
            return {self.input_name: data}
        else:
            return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)

qdr = QuantizationDataReader(calib_ds, batch_size=64, input_name=ort_session.get_inputs()[0].name)

q_static_opts = {"ActivationSymmetric":False,
                 "WeightSymmetric":True}

quantized_model = quantize_static(model_input='calc_model_quant_static_prep.onnx',
                                               model_output='calc_model_quant_static.onnx',
                                               calibration_data_reader=qdr,
                                               extra_options=q_static_opts)

# Load the static quantized model
ort_session_quant_static = onnxruntime.InferenceSession('calc_model_quant_static.onnx')


### Run Models ###

def run_onnx_model(dataloader, ort_session, input_name):
    features_quant_dynamic = []

    for inputs in dataloader:
        # Convert the tensor to numpy
        inputs = inputs.detach().cpu().numpy()

        # Create the input dictionary
        ort_input = {input_name: inputs}

        # Run the model
        ort_output = ort_session.run(None, ort_input)

        # Append the output to the list
        features_quant_dynamic.append(ort_output[0])

    features_quant_dynamic = torch.from_numpy(np.concatenate(features_quant_dynamic, axis=0))
    return features_quant_dynamic

# Check if model is a valid ONNX model
onnx_model_quant_static = onnx.load("calc_model_quant_static.onnx")
onnx.checker.check_model(onnx_model_quant_static)

# Load the ONNX model
ort_session_quant_static = onnxruntime.InferenceSession("calc_model_quant_static.onnx")

input_name = ort_session_quant_static.get_inputs()[0].name

# Process database images
db_features_quant_static = run_onnx_model(db_dataloader, ort_session_quant_static, input_name)
print(db_features_quant_static.shape)

# Process query images
q_features_quant_static = run_onnx_model(q_dataloader, ort_session_quant_static, input_name)
print(q_features_quant_static.shape)


### Average Time ###

def measure_time_onnx(dataloader, ort_session, input_name, iterations, desc):
    
    start_time = time.time()

    for _ in tqdm(range(iterations), desc=desc):

        run_onnx_model(dataloader, ort_session, input_name)

    end_time = time.time()

    avg_time = (end_time - start_time)  / iterations

    return avg_time

db_time_quant_static = measure_time_onnx(db_dataloader, ort_session_quant_static, input_name, ITERATIONS, "Processing database dataset")
print(f"Database Average Time: {db_time_quant_static}")

q_time_quant_static = measure_time_onnx(q_dataloader, ort_session_quant_static, input_name, ITERATIONS, "Processing query dataset")
print(f"Query Average Time: {q_time_quant_static}")


### Evaluation ###

# Convert the predictions to a numpy array
features = db_features_quant_static.detach().cpu().numpy()

# Get the predicted labels (assuming a binary classification problem)
predicted_labels = (features > 0.9).astype(int)

# Get the ground truth labels
true_labels = gt_hard

# Get the predicted labels
predicted_labels = features.argmax(axis=1)

# Calculate the accuracy
accuracy = (predicted_labels == true_labels).mean()
print('Accuracy:', accuracy)

similarity_matrix = cosine_similarity(db_features_quant_static.detach().numpy(), q_features_quant_static.detach().numpy())

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
    f.write(f"Database Average Time: {db_time_quant_static}\n")
    f.write(f"Query Average Time: {q_time_quant_static}\n")
