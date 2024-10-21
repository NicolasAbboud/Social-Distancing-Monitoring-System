# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .utils import PriorBox
from ..box_utils import batched_decode


class FEM(nn.Module):
    """
    Feature Enhancement Module (FEM) for enhancing feature maps.
    
    Args:
        channel_size (int): Number of input channels.
    """
    def __init__(self, channel_size):
        super(FEM, self).__init__()
        self.cs = channel_size
        self.cpm1 = nn.Conv2d(self.cs, 256, kernel_size=3, padding=1)
        self.cpm2 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=2, padding=2)
        self.cpm3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.cpm4 = nn.Conv2d(256, 128, kernel_size=3, dilation=2,  padding=2)
        self.cpm5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass through the FEM.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Concatenated output of the FEM.
        """
        x1_1 = self.cpm1(x).relu()
        x1_2 = self.cpm2(x).relu()
        x2_1 = self.cpm3(x1_2).relu()
        x2_2 = self.cpm4(x1_2).relu()
        x3_1 = self.cpm5(x2_2).relu()
        return torch.cat([x1_1, x2_1, x3_1], dim=1)


class SSD(nn.Module):
    """
    Single Shot Multibox Architecture (SSD) for object detection.
    
    The network is composed of a base ResNet network followed by additional
    convolutional layers for multibox detection. Each multibox layer branches into:
        1) conv2d for class confidence scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding boxes specific to the layer's feature map size.
    
    Args:
        cfg (dict): Configuration dictionary.
    """

    def __init__(self, cfg):
        super(SSD, self).__init__()
        self.num_classes = 2 # Background and face
        self.cfg = cfg
        
        # Initialize ResNet-152 as the base network
        resnet = torchvision.models.resnet152(pretrained=False)
        self.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        self.layer5 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        output_channels = [256, 512, 1024, 2048, 512, 256]

        # Feature Pyramid Network (FPN) layers
        fpn_in = output_channels

        self.latlayer3 = nn.Conv2d(fpn_in[3], fpn_in[2], kernel_size=1)
        self.latlayer2 = nn.Conv2d(fpn_in[2], fpn_in[1], kernel_size=1)
        self.latlayer1 = nn.Conv2d(fpn_in[1], fpn_in[0], kernel_size=1)

        self.smooth3 = nn.Conv2d(fpn_in[2], fpn_in[2], kernel_size=1)
        self.smooth2 = nn.Conv2d(fpn_in[1], fpn_in[1], kernel_size=1)
        self.smooth1 = nn.Conv2d(fpn_in[0], fpn_in[0], kernel_size=1)

        # Feature Enhancement Modules (FEM)
        cpm_in = output_channels
        self.cpm3_3 = FEM(cpm_in[0])
        self.cpm4_3 = FEM(cpm_in[1])
        self.cpm5_3 = FEM(cpm_in[2])
        self.cpm7 = FEM(cpm_in[3])
        self.cpm6_2 = FEM(cpm_in[4])
        self.cpm7_2 = FEM(cpm_in[5])
        
        # Multibox head for localization and confidence predictions
        head = pa_multibox(output_channels, self.cfg['mbox'], self.num_classes)  
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        # Softmax layer for testing
        self.softmax = nn.Softmax(dim=-1)

        # Cache to stop computing new priors per forward pass
        self.prior_cache = {
        }

    def init_priors(self, feature_maps, image_size):
        """
        Initialize prior boxes for the given feature maps and image size.
    
        Args:
            feature_maps (list): List of feature map sizes.
            image_size (list): Size of the input image.
    
        Returns:
            torch.Tensor: Prior boxes.
        """

        # Create a unique key based on feature maps and image size
        key = ".".join([str(item) for i in range(len(feature_maps)) for item in feature_maps[i]]) + \
              "," + ".".join([str(_) for _ in image_size])
        if key in self.prior_cache:
            return self.prior_cache[key].clone()
        
        # Generate prior boxes
        priorbox = PriorBox(self.cfg, image_size, feature_maps)
        prior = priorbox.forward()
        self.prior_cache[key] = prior.clone()
        return prior


    def forward(self, x, confidence_threshold, nms_threshold):
        """
        Applies network layers and operations on input image(s) x.
    
        Args:
            x (torch.Tensor): Input image or batch of images. Shape: [batch, 3, 300, 300].
            confidence_threshold (float): Confidence threshold for detections.
            nms_threshold (float): Non-maximum suppression threshold.
    
        Returns:
            torch.Tensor: Output class label predictions, confidence scores, and location predictions.
        """
        image_size = [x.shape[2], x.shape[3]]
        loc = list()
        conf = list()

        # Pass through ResNet152 layers
        conv3_3_x = self.layer1(x)
        conv4_3_x = self.layer2(conv3_3_x)
        conv5_3_x = self.layer3(conv4_3_x)
        fc7_x = self.layer4(conv5_3_x)
        conv6_2_x = self.layer5(fc7_x)
        conv7_2_x = self.layer6(conv6_2_x)

        # Feature Pyramid Network (FPN)             
        lfpn3 = self._upsample_product(
            self.latlayer3(fc7_x), self.smooth3(conv5_3_x))
        lfpn2 = self._upsample_product(
            self.latlayer2(lfpn3), self.smooth2(conv4_3_x))
        lfpn1 = self._upsample_product(
            self.latlayer1(lfpn2), self.smooth1(conv3_3_x))

        conv5_3_x = lfpn3
        conv4_3_x = lfpn2
        conv3_3_x = lfpn1
        
        # Collect feature maps from different layers
        sources = [
            self.cpm3_3(conv3_3_x),
            self.cpm4_3(conv4_3_x),
            self.cpm5_3(conv5_3_x),
            self.cpm7(fc7_x),
            self.cpm6_2(conv6_2_x),
            self.cpm7_2(conv7_2_x)]
        
        
        # Apply multibox head to source layers
        featuremap_size = []
        for (x, l, c) in zip(sources, self.loc, self.conf):
            featuremap_size.append([x.shape[2], x.shape[3]])
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())

            # Max in out
            len_conf = len(conf)
            out = self.mio_module(c(x), len_conf)

            conf.append(out.permute(0, 2, 3, 1).contiguous())

        # Progressive Anchor
        mbox_num = self.cfg['mbox'][0]
        face_loc = torch.cat([
            o[:, :, :, :4*mbox_num].contiguous().view(o.size(0), -1)
            for o in loc], dim=1)
        face_conf = torch.cat([
            o[:, :, :, :2*mbox_num].contiguous().view(o.size(0), -1)
            for o in conf], dim=1)

        # Test Phase
        self.priors = self.init_priors(featuremap_size, image_size)
        self.priors = self.priors.to(face_conf.device)
        conf_preds = face_conf.view(
            face_conf.size(0), -1, self.num_classes).softmax(dim=-1)
        face_loc = face_loc.view(face_loc.size(0), -1, 4)
        boxes = batched_decode(
            face_loc, self.priors,
            self.cfg["variance"]
        )
        scores = conf_preds.view(-1, self.priors.shape[0], 2)[:, :, 1:]
        output = torch.cat((boxes, scores), dim=-1)
        return output

    def mio_module(self, each_mmbox, len_conf):
        """
        Apply Max-In-Out (MIO) module to the multibox head outputs.
    
        Args:
            each_mmbox (torch.Tensor): Multibox head output.
            len_conf (int): Length of the confidence list.
    
        Returns:
            torch.Tensor: Output after applying MIO module.
        """
        chunk = torch.chunk(each_mmbox, each_mmbox.shape[1], 1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        if len_conf == 0:
            out = torch.cat([bmax, chunk[3]], dim=1)
        else:
            out = torch.cat([chunk[3], bmax], dim=1)
        if len(chunk) == 6:
            out = torch.cat([out, chunk[4], chunk[5]], dim=1)
        elif len(chunk) == 8:
            out = torch.cat(
                [out, chunk[4], chunk[5], chunk[6], chunk[7]], dim=1)
        return out

    def _upsample_product(self, x, y):
        """
        Upsample and add two feature maps.
    
        Args:
            x (torch.Tensor): Top feature map to be upsampled.
            y (torch.Tensor): Lateral feature map.
    
        Returns:
            torch.Tensor: Added feature map.
    
        Note:
            In PyTorch, when input size is odd, the upsampled feature map
            with `F.upsample(..., scale_factor=2, mode='nearest')`
            may not equal the lateral feature map size.
            So we choose bilinear upsample which supports arbitrary output sizes.
        """
        return y * F.interpolate(
            x, size=y.shape[2:], mode="bilinear", align_corners=True)


class DeepHeadModule(nn.Module):
    """
    Deep Head Module for localization and confidence predictions.
    
    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._mid_channels = min(self._input_channels, 256)

        self.conv1 = nn.Conv2d(
            self._input_channels, self._mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            self._mid_channels, self._mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            self._mid_channels, self._mid_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            self._mid_channels, self._output_channels, kernel_size=1,)

    def forward(self, x):
        """
        Forward pass through the Deep Head Module.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.conv1(x).relu()
        out = self.conv2(out).relu()
        out = self.conv3(out).relu()
        out = self.conv4(out)
        return out


def pa_multibox(output_channels, mbox_cfg, num_classes):
    """
    Create localization and confidence layers for the multibox head.
    
    Args:
        output_channels (list): List of output channels for each layer.
        mbox_cfg (list): Configuration for the number of boxes per layer.
        num_classes (int): Number of classes.
    
    Returns:
        tuple: Localization and confidence layers.
    """
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        input_channels = 512
        if k == 0:
            loc_output = 4
            conf_output = 2
        elif k == 1:
            loc_output = 8
            conf_output = 4
        else:
            loc_output = 12
            conf_output = 6
        loc_layers += [
            DeepHeadModule(input_channels, mbox_cfg[k] * loc_output)]
        conf_layers += [
            DeepHeadModule(input_channels, mbox_cfg[k] * (2+conf_output))]
    return (loc_layers, conf_layers)


