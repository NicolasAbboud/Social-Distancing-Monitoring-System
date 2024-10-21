# Import necessary libraries
import numpy as np
import torch


def to_cuda(elements, device):
    """
    Move elements to a CUDA device if available.
    
    Args:
        elements (torch.Tensor or list/tuple of torch.Tensors): Elements to move to the device.
        device (torch.device): The target device.
    
    Returns:
        torch.Tensor or list/tuple of torch.Tensors: Elements moved to the device.
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.to(device) for x in elements]
        return elements.to(device)
    return elements


def get_device():
    """
    Get the appropriate device (CUDA if available, otherwise CPU).
    
    Returns:
        torch.device: The target device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def image_to_torch(image, device):
    """
    Convert an image to a torch tensor and move it to the specified device.
    
    Args:
        image (np.ndarray): Input image in HWC format.
        device (torch.device): The target device.
    
    Returns:
        torch.Tensor: Image converted to a torch tensor and moved to the device.
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    else:
        assert image.dtype == np.float32
    image = np.rollaxis(image, 2)
    image = image[None, :, :, :]
    image = torch.from_numpy(image).to(device)
    return image
