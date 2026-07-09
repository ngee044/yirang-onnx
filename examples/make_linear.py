import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

X = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 4])
Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3])

rng = np.random.default_rng(0)
W = numpy_helper.from_array(rng.standard_normal((4, 3)).astype(np.float32), name="W")
b = numpy_helper.from_array(rng.standard_normal((3,)).astype(np.float32), name="b")

node = helper.make_node("Gemm", ["input", "W", "b"], ["output"], name="linear")
graph = helper.make_graph([node], "linear_graph", [X], [Y], initializer=[W, b])

model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
model.ir_version = 9
model.producer_name = "hands-on"
onnx.checker.check_model(model)

import os
os.makedirs("models", exist_ok=True)
onnx.save(model, "models/linear.onnx")
print("wrote models/linear.onnx")
