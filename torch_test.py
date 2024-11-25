import torch

if torch.cuda.is_available():
    device_properties = torch.cuda.get_device_properties(0)
    print(f"GPU Name: {device_properties.name}")
    print(f"Total Memory (GB): {device_properties.total_memory / 1e9}")
    print(f"CUDA Capability (Major.Minor): {device_properties.major}.{device_properties.minor}")
else:
    print("CUDA is not available.")


# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Get the current device name
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA device name: {device_name}")

    # Set the device
    device = torch.device('cuda')
else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device('cpu')

# Create a random tensor on the CPU
x_cpu = torch.randn(3, 3)
print("Tensor on CPU:")
print(x_cpu)

# Move tensor to the appropriate device (GPU if available)
x = x_cpu.to(device)
print(f"Tensor moved to {device}:")
print(x)

# Perform a simple operation
y = x * 2
print(f"Result of tensor operation on {device}:")
print(y)

# Move the result back to CPU (if it was on GPU)
y_cpu = y.to('cpu')
print("Result moved back to CPU:")
print(y_cpu)
