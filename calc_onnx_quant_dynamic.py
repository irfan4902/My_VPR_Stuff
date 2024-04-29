# %% [markdown]
# # Import Stuff

# # %%
# %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# %pip install opencv-python numpy seaborn matplotlib scikit-learn ipykernel tqdm pillow
# %pip install onnx onnxruntime quanto

# %%
import os
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data

from datasets import dataset_utils
from matching import matching
from evaluation.metrics import createPR, recallAt100precision, recallAtK
from datasets.load_dataset import GardensPointDataset, SFUDataset, StLuciaDataset

# %%
# GardensPointDataset().load()
# SFUDataset().load()
# StLuciaDataset().load()

# %% [markdown]
# # Constants

# %%
WEIGHTS_FILE = "calc.caffemodel.pt"
DATASET = "SFU"
DATABASE_FOLDER = "dry"
QUERY_FOLDER = "jan"
ITERATIONS = 5 # for testing average duration
BATCH_SIZE = 8
NUM_WORKERS = 4

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

# %%
dataset_db = CustomImageDataset(f"images/{DATASET}", DATABASE_FOLDER, preprocess)
dataset_q = CustomImageDataset(f"images/{DATASET}", QUERY_FOLDER, preprocess)
gt_data = np.load(f'images/{DATASET}/GT.npz')
gt_hard = gt_data['GThard']
gt_soft = gt_data['GTsoft']

print("Dataset Length:", len(dataset_db))
dataset_db[0]

# %%
db_dataloader = DataLoader(dataset_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
q_dataloader = DataLoader(dataset_q, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# %% [markdown]
# # Model Definition

# %%
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

# %% [markdown]
# ### Dynamic Quantization (ONNX)

# %%
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'calc_model.onnx'
model_quant = 'calc_model_quant_dynamic.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

# Load the dynamic quantized model
ort_session_quant_dynamic = onnxruntime.InferenceSession("calc_model_quant_dynamic.onnx")

# %% [markdown]
# # Run Models

# %%
def run_onnx_model(dataloader, ort_session, input_name):
    features = []

    for inputs in dataloader:
        # Convert the tensor to numpy
        inputs = inputs.detach().cpu().numpy()

        # Create the input dictionary
        ort_input = {input_name: inputs}

        # Run the model
        ort_output = ort_session.run(None, ort_input)

        # Append the output to the list
        features.append(ort_output[0])

    features = torch.from_numpy(np.concatenate(features, axis=0))
    return features


# %% [markdown]
# ### Dynamic Quantization (ONNX)


# Check if model is a valid ONNX model
onnx_model_quant_dynamic = onnx.load("calc_model_quant_dynamic.onnx")
onnx.checker.check_model(onnx_model_quant_dynamic)

# Load the ONNX model
ort_session_quant_dynamic = onnxruntime.InferenceSession("calc_model_quant_dynamic.onnx")

input_name = ort_session_quant_dynamic.get_inputs()[0].name


# Process database images
db_features_quant_dynamic = run_onnx_model(db_dataloader, ort_session_quant_dynamic, input_name)
print(db_features_quant_dynamic.shape)

# Process query images
q_features_quant_dynamic = run_onnx_model(q_dataloader, ort_session_quant_dynamic, input_name)
print(q_features_quant_dynamic.shape)


# %% [markdown]
# # Average Time

def measure_time_onnx(dataloader, ort_session, input_name, iterations, desc):
    
    start_time = time.time()

    for _ in tqdm(range(iterations), desc=desc):

        run_onnx_model(dataloader, ort_session, input_name)

    end_time = time.time()

    avg_time = (end_time - start_time)  / iterations

    return avg_time


# %% [markdown]
# ### Normal Model

db_time_quant_dynamic = measure_time_onnx(db_dataloader, ort_session_quant_dynamic, input_name, ITERATIONS, "Processing database dataset")
print(f"Database Average Time: {db_time_quant_dynamic}")

q_time_quant_dynamic = measure_time_onnx(q_dataloader, ort_session_quant_dynamic, input_name, ITERATIONS, "Processing query dataset")
print(f"Query Average Time: {q_time_quant_dynamic}")

# %% [markdown]
# # Evaluation

# %%
# parameter_count = sum(p.numel() for p in model_quant.parameters())
# print(f"Total Parameters: {parameter_count}")

# memory_size = parameter_count * 4

# for unit in ['KB', 'MB', 'GB']:
#     if memory_size < 1024:
#         print(f"Memory Size: {memory_size:.2f} {unit}")
#         break
#     memory_size /= 1024

# %%
# Convert the predictions to a numpy array
bruh = db_features_quant_dynamic.detach().cpu().numpy()

# Get the predicted labels (assuming a binary classification problem)
predicted_labels = (bruh > 0.9).astype(int)

# Get the ground truth labels
true_labels = gt_data['GThard']

# Get the predicted labels
predicted_labels = bruh.argmax(axis=1)

# Calculate the accuracy
accuracy = (predicted_labels == true_labels).mean()
print('Accuracy:', accuracy)

# %%
similarity_matrix = cosine_similarity(db_features_quant_dynamic.detach().numpy(), q_features_quant_dynamic.detach().numpy())

# plt.figure()
# sns.heatmap(similarity_matrix, cmap='viridis')
# plt.title('Similarity Matrix')
# plt.axis('off')
# plt.show()


# %%
# best matching per query in S for single-best-match VPR
M1 = matching.best_match_per_query(similarity_matrix)

# find matches with S>=thresh using an auto-tuned threshold for multi-match VPR
M2 = matching.thresholding(similarity_matrix, 0.92)

# # show M's
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.imshow(M1)
# ax1.axis('off')
# ax1.set_title('Best match per query')
# ax2 = fig.add_subplot(122)
# ax2.imshow(M2)
# ax2.axis('off')
# ax2.set_title('Thresholding S>=thresh')

# %%
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

# %%
# Open the file in write mode
with open('data_quant_dynamic.txt', 'w') as f:
    # Write the database and query times to the file
    f.write(f"Database Average Time: {db_time_quant_dynamic}\n")
    f.write(f"Query Average Time: {q_time_quant_dynamic}\n")
    # f.write(f"Parameter count: {parameter_count}\n")

# %%
# precision-recall curve
P, R = createPR(similarity_matrix, gt_hard, gt_soft, matching='multi', n_thresh=100)
plt.figure()
plt.plot(R, P)
plt.xlim(0, 1), plt.ylim(0, 1.01)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Result on {DATASET} {DATABASE_FOLDER}--{QUERY_FOLDER}')
plt.grid('on')
plt.draw()
plt.show()


# %%
# # Plot recall against memory consumption
# plt.figure(figsize=(10, 6))
# plt.plot(memory_size, recall, marker='o')
# plt.title('Recall vs Memory Consumption')
# plt.xlabel('Memory Consumption')
# plt.ylabel('Recall')
# plt.show()

# # Plot accuracy against memory consumption
# plt.figure(figsize=(10, 6))
# plt.plot(memory_size, a, marker='o')
# plt.title('Accuracy vs Memory Consumption')
# plt.xlabel('Memory Consumption')
# plt.ylabel('Accuracy')
# plt.show()

# # Plot recall against parameter count
# plt.figure(figsize=(10, 6))
# plt.plot(parameter_count, recall, marker='o')
# plt.title('Recall vs Parameter Count')
# plt.xlabel('Parameter Count')
# plt.ylabel('Recall')
# plt.show()
