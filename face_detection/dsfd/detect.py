# Import necessary libraries
import torch
import numpy as np
import typing
from .face_ssd import SSD
from .config import resnet152_model_config
from .. import torch_utils
from ..base import Detector
from ..build import DETECTOR_REGISTRY

# Path to the model weights
model_path = "../../models/WIDERFace_DSFD_RES152.pth"

# Register the DSFDDetector class in the DETECTOR_REGISTRY
@DETECTOR_REGISTRY.register_module
class DSFDDetector(Detector):

    def __init__(self, *args, **kwargs):
        """
        Initialize the DSFDDetector with the given arguments.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        state_dict = torch.load(
            model_path,
            map_location=self.device,
            )

        self.net = SSD(resnet152_model_config)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.net = self.net.to(self.device)

    @torch.no_grad()
    def _detect(self, x: torch.Tensor,) -> typing.List[np.ndarray]:
        """
        Perform batched detection on the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, 3, H, W].
        
        Returns:
            list: List of length N with shape [num_boxes, 5] per element.
        """

        # Convert RGB to BGR
        x = x[:, [2, 1, 0], :, :]

        # Perform detection with mixed precision if fp16_inference is enabled
        with torch.cuda.amp.autocast(enabled=self.fp16_inference):
            boxes = self.net(
                x, self.confidence_threshold, self.nms_iou_threshold
            )
        return boxes
