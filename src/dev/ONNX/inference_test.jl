using ONNX
using Flux

# Load the model
input_data = rand(Float32, 1, 15, 14)
model = ONNX.load("src/dev/ONNX/model.onnx", input_data)
flux_model = ONNX.compile(model)

# Use the model
prediction = flux_model(input_data)
