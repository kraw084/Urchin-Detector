import torch

print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")