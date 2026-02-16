import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 1. Load the model
try:
    with open('model.p', 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: model.p not found!")

# 2. Define the input type (21 landmarks * 3 coordinates = 63 features)
initial_type = [('float_input', FloatTensorType([None, 63]))]

# 3. Convert to ONNX
onx = convert_sklearn(model, initial_types=initial_type)

# 4. Save with the CORRECT capitalization
with open("sign_model.onnx", "wb") as f:
    f.write(onx.SerializeToString()) # <--- Capitalized 'S' and 'T'

print("🚀 Success! 'sign_model.onnx' created and saved.")