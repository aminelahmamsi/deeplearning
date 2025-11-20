import torch
try:
    import torch_directml
except ImportError:
    directml_loaded = False


# Check if CUDA (NVIDIA GPU) is available
def get_device(no_print = True):
    """
    Detects and returns the most suitable computation device.

    Priority order:
        1. CUDA (NVIDIA GPU)
        2. DirectML (e.g., AMD GPU, if available)
        3. CPU (fallback)

    Args:
        no_print (bool, optional): If False, prints which device is being used. Default is True.

    Returns:
        torch.device: The selected computation device.
    """
    # Check if CUDA is available (for Nvidia GPUs or other supported devices)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if (not no_print):
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    
    elif (directml_loaded):
        # Check if DirectML is available (for AMD GPUs or other supported devices)
        try: 
            device = torch_directml.device()
            if (not no_print):
                print("Using DirectML device")
        # If DirectML is not available, fall back to CPU
        except: 
            device = torch.device("cpu")
            if (not no_print):
                print("Using CPU")
    else:
        device = torch.device("cpu")
        if (not no_print):
            print("Using CPU")
            
    return device