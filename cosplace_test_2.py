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
    model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
    model.eval()

    print(model)

    model_class = type(model)

    #
    # Quantize Model
    #
    # # Doesn't work?
    # quantized_model = torch.quantization.quantize_dynamic(
    #     model, {torch.nn.Linear}, dtype=torch.qint8
    # )
    # print(quantized_model)
    # torch.save(quantized_model.state_dict(), "quantized_model.pth")
    # quantized_model = model_class(backbone="ResNet50", fc_output_dim=2048)  # replace with your model class
    # quantized_model.load_state_dict(torch.load("quantized_model.pth"), strict=False)

    # dummy_input = torch.randn(BATCH_SIZE, 3, 480, 752) 
    # torch.onnx.export(model, dummy_input, "model.onnx")
    # from onnxruntime.quantization import quantize, QuantizationMode
    # quantized_model = quantize("model.onnx", "quantized_model.onnx", quant)

    # Trace the model
    dummy_input = torch.randn(BATCH_SIZE, 3, 480, 752, requires_grad=True)
    dummy_out = model(dummy_input)
    print(dummy_out.shape)

    # Convert to ONNX
    model_fp32_path = 'cosplace_fp32.onnx'
    torch.onnx.export(model,                                            # model
                      dummy_input,                                      # model input
                      model_fp32_path,                                  # path
                      export_params=True,                               # store the trained parameter weights inside the model file
                      opset_version=14,                                 # the ONNX version to export the model to
                      do_constant_folding=True,                         # constant folding for optimization
                      input_names = ['input'],                          # input names
                      output_names = ['output'],                        # output names
                      dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                                    'output' : {0 : 'batch_size'}})

    # Verify the ONNX Model
    model_onnx = onnx.load(model_fp32_path)
    onnx.checker.check_model(model_onnx)

    # PyTorch Tensor to NumPy Array
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # Prepare the Models
    ort_provider = ['CPUExecutionProvider']
    ort_sess = onnxruntime.InferenceSession(model_fp32_path, providers=ort_provider)

    # Prepare for quantization
    model_prep_path = 'cosplace_prep.onnx'
    onnxruntime.quantization.shape_inference.quant_pre_process(model_fp32_path, model_prep_path, skip_symbolic_shape=False)

    class QuantizationDataReader(quantization.CalibrationDataReader):
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
                return {self.input_name: self.to_numpy(batch[0])}
            else:
                return None

        def rewind(self):
            self.enum_data = iter(self.torch_dl)

    qdr = QuantizationDataReader(calib_ds, batch_size=64, input_name=ort_sess.get_inputs()[0].name)

    # Quantize the Model
    q_static_opts = {"ActivationSymmetric":False,
                 "WeightSymmetric":True}

    model_int8_path = 'cosplace_int8.onnx'
    quantized_model = onnxruntime.quantization.quantize_static(model_input=model_prep_path,
                                                model_output=model_int8_path,
                                                calibration_data_reader=qdr,
                                                extra_options=q_static_opts)

    print(quantized_model)

    #
    # Run Model
    #
  
    # # Pass database tensor through the model
    # start_time = time.time()

    # db_features = []

    # with torch.no_grad():

    #     for batch in tqdm(db_dataloader):
    #         output = model(batch)
    #         db_features.append(output)

    # db_features = torch.cat(db_features, axis=0)

    # print(db_features.shape)

    # end_time = time.time()

    # print("Database Duration:", end_time - start_time)

    # # Pass query tensor through the model
    # start_time = time.time()

    # q_features = []

    # with torch.no_grad():

    #     for batch in tqdm(q_dataloader):
    #         output = model(batch)
    #         q_features.append(output)

    # q_features = torch.cat(q_features, axis=0)

    # print(q_features.shape)

    # end_time = time.time()

    # print("Query Duration:", end_time - start_time)


    # #
    # # Run Quantized Model
    # #

    # quantized_model.eval()

    # # Pass database tensor through the model
    # start_time = time.time()

    # db_features = []

    # with torch.no_grad():

    #     for batch in tqdm(db_dataloader):
    #         output = quantized_model(batch)
    #         db_features.append(output)

    # db_features = torch.cat(db_features, axis=0)

    # print(db_features.shape)

    # end_time = time.time()

    # print("Database Duration:", end_time - start_time)

    # # Pass query tensor through the model
    # start_time = time.time()

    # q_features = []

    # with torch.no_grad():

    #     for batch in tqdm(q_dataloader):
    #         output = quantized_model(batch)
    #         q_features.append(output)

    # q_features = torch.cat(q_features, axis=0)

    # print(q_features.shape)

    # end_time = time.time()

    # print("Query Duration:", end_time - start_time)