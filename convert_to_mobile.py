import pickle
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import tensorflow as tf

# 1. Load your Python model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# 2. Convert Scikit-Learn model to ONNX format
initial_type = [('float_input', FloatTensorType([None, 63]))] # 21 landmarks * 3 (x,y,z)
onx = convert_sklearn(model, initial_types=initial_type)

with open("sign_model.onnx", "wb") as f:
    f.write(onx.serialize_to_string())

print("✅ Step 1: Model converted to ONNX!")

# 3. Note for Mobile Integration
print("\n--- NEXT STEPS ---")
print("We now have 'sign_model.onnx'. In our Java project,")
print("we will use the ONNX Runtime for Android or convert")
print("this to a .tflite file for the neural network.")