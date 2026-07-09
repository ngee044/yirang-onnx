import os
import numpy as np
import onnx
from onnx import numpy_helper

arr = np.array([[1, 2, 3, 4]], dtype=np.float32)
tensor = numpy_helper.from_array(arr, name="input")
os.makedirs("tensors", exist_ok=True)
with open("tensors/input.pb", "wb") as f:
    f.write(tensor.SerializeToString())
print("wrote tensors/input.pb")
