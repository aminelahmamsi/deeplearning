### `get_device(no_print=True)`

Detects and returns the most suitable computation device for PyTorch.

#### Priority:
1. **CUDA** (NVIDIA GPU)
2. **DirectML** (e.g., AMD GPU, if supported)
3. **CPU** (fallback option)

#### Parameters:
- `no_print` (`bool`, optional):  
  If set to `False`, prints which device is being used. Default is `True`.

#### Returns:
- `torch.device`:  
  The selected device to run computations on.
