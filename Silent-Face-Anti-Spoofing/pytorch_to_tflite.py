"""
Code for generating tflite model (pytorch -> onnx -> tensorflow -> tflite) for coding task.
Please run in Silent-Face-Anti-Spoofing folder.
Please run generate_tflie_test_images.py first to create test image.
This is done to ensure model performance does not decay for a real test case during conversions.
"""

import os
import shutil
import torch
import numpy as np
import onnx
import onnxruntime
from onnx_tf.backend import prepare
import tensorflow as tf
import cv2
from src.anti_spoof_predict import AntiSpoofPredict
from src.data_io import transform as trans


# paths
PYTORCH_MODEL_PATH = "./resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
ONNX_MODEL_PATH = "./resources/onnx_anti_spoof_models/2.7_80x80_MiniFASNetV2.onnx"
TF_MODEL_PATH = "./resources/tf_anti_spoof_models/2.7_80x80_MiniFASNetV2"
TFLIFE_MODEL_PATH = "./resources/tflite_anti_spoof_models/2.7_80x80_MiniFASNetV2.tflite"

if not os.path.exists("./resources/onnx_anti_spoof_models"):
    os.mkdir("./resources/onnx_anti_spoof_models")
if not os.path.exists("./resources/tf_anti_spoof_models"):
    os.mkdir("./resources/tf_anti_spoof_models")
if not os.path.exists("./resources/tflite_anti_spoof_models"):
    os.mkdir("./resources/tflite_anti_spoof_models")

SAMPLE_IMAGE_PATH = "./images/sample/"

# Helper functions from converting pytorch tensor to numpy array
def pytorch_tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# -- Load and run inference on pytorch model --

# Bit hacky passing in device id here, done to not have to change their code
antispoof_model = AntiSpoofPredict(0)
antispoof_model._load_model(PYTORCH_MODEL_PATH)
pytorch_model = antispoof_model.model

# Make sure model is in eval mode for dropout etc
pytorch_model.eval()

img = cv2.imread("../test/fixtures/image_F1.bmp")
test_transform = trans.Compose([
            trans.ToTensor(),
        ])
sample_input_image = test_transform(img)
sample_input_image = sample_input_image.unsqueeze(0)

torch_out = pytorch_model(sample_input_image)


# -- Export Pytorch model to onnx --

torch.onnx.export(
    pytorch_model,  # PyTorch Model
    sample_input_image,  # Input tensor
    ONNX_MODEL_PATH,  # Output file (eg. 'output_model.onnx')
    opset_version=16,  # Operator support version
    input_names=['input'],  # Input tensor name (arbitary)
    output_names=['output'],  # Output tensor name (arbitary)
)

# Check that onnx model is valid
onnx_model = onnx.load(ONNX_MODEL_PATH)
onnx.checker.check_model(onnx_model)

# Check that model performs similar in onnx as pytorch model

ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)

# compute ONNX Runtime output prediction
onnx_input = {ort_session.get_inputs()[0].name: pytorch_tensor_to_numpy(sample_input_image)}
onnx_out = ort_session.run(None, onnx_input)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(pytorch_tensor_to_numpy(torch_out), onnx_out[0], rtol=1e-03, atol=1e-05)

print("ONNX model passed test")


# -- Export onnx model to tensorflow --

# tensorflow represenation of onnx model
tf_rep = prepare(onnx_model, device="cpu")

tf_rep.export_graph(TF_MODEL_PATH)

# Test tensorflow model

model = tf.saved_model.load(TF_MODEL_PATH)
model.trainable = True

sample_input_image = tf.convert_to_tensor(sample_input_image, dtype=tf.float32)
tf_out = model(input=sample_input_image)

# Compare PyTorch and Tensorflow results
np.testing.assert_allclose(tf_out["output"].numpy(), pytorch_tensor_to_numpy(torch_out), rtol=1e-03, atol=1e-05)
print("Tensorflow model passed test")


# -- Export tensorflow model to tflite --

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL_PATH)
tflite_model = converter.convert()

# Save the model
with open(TFLIFE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=TFLIFE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
tflite_input = interpreter.get_input_details()
tflite_output = interpreter.get_output_details()

# Test the model on random input data
input_shape = tflite_input[0]['shape']
interpreter.set_tensor(tflite_input[0]['index'], sample_input_image)

interpreter.invoke()

# Get results from tflite output buffer
tflite_results = interpreter.get_tensor(tflite_output[0]['index'])

# Compare PyTorch and tflite results
np.testing.assert_allclose(tflite_results, pytorch_tensor_to_numpy(torch_out), rtol=1e-03, atol=1e-05)
print("Tflite model passed test\n")

# Print results for all models
print(f"Torch model score: {pytorch_tensor_to_numpy(torch_out)}")
print(f"ONNX model score: {onnx_out[0]}")
print(f"TF model score: {tf_out['output'].numpy()}")
print(f"TFLite model score: {tflite_results}\n")

# Copy tflite model file to challenge assets/models folder
shutil.copy(TFLIFE_MODEL_PATH, "../assets/models/")
print("Copied tflite model to ../assets/models")
