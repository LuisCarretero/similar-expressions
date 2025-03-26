# using ONNX
# using Flux

# # Load the model
# input_data = rand(Float32, 1, 15, 14)
# model = ONNX.load("src/dev/ONNX/model-9j0cbuui.onnx", input_data)
# flux_model = ONNX.compile(model)

# # Use the model
# prediction = flux_model(input_data)


import ONNXRunTime as ORT

input_data = rand(Float32, 1, 15, 14)

run_id = "fhgrred2"
model = ORT.load_inference("src/dev/ONNX/onnx-models/model-$(run_id).onnx")
input = Dict("onnx::Flatten_0" => input_data)
output = model(input)

