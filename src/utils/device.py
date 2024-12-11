"""
graphics device checker
"""

import torch

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()

def get_device():
    if cuda_available:
        return torch.device('cuda')
    elif mps_available:
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_device_count():
    if cuda_available:
        return torch.cuda.device_count()
    elif mps_available:
        return torch.mps.device_count()
    else:
        return torch.cpu.device_count()


# if __name__ == "__main__":
#     device = get_device()
#     device_count = get_device_count()
#     print(f"Using device: {device}:{device_count}")

