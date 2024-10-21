# Import necessary libraries
import torch
import math


class PriorBox(object):
    """
    Compute prior box coordinates in center-offset form for each source feature map.
    
    Args:
        cfg (dict): Configuration dictionary containing parameters for prior box generation.
        image_size (list): Size of the input image [height, width].
        feature_maps (list): List of feature map sizes.
    """

    def __init__(self, cfg, image_size, feature_maps):
        super(PriorBox, self).__init__()
        self.image_size = image_size
        self.feature_maps = feature_maps
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.min_sizes = cfg["min_sizes"]

        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']

        # Ensure variances are greater than 0
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        """
        Compute the prior boxes for each feature map.
        
        This method generates the prior boxes in center-offset form for each
        feature map location based on the configuration provided during initialization.
        
        Returns:
            torch.Tensor: Prior boxes in center-offset form with shape [num_priors, 4].
        """
        mean = []
        
        # Adjust feature maps and steps based on the number of min_sizes
        if len(self.min_sizes) == 5:
            self.feature_maps = self.feature_maps[1:]
            self.steps = self.steps[1:]
        if len(self.min_sizes) == 4:
            self.feature_maps = self.feature_maps[2:]
            self.steps = self.steps[2:]

        for k, f in enumerate(self.feature_maps):
            for i in range(f[0]):
                for j in range(f[1]):

                    f_k_i = self.image_size[0] / self.steps[k]
                    f_k_j = self.image_size[1] / self.steps[k]

                    # Unit center x,y
                    cx = (j + 0.5) / f_k_j
                    cy = (i + 0.5) / f_k_i

                    # Aspect ratio: 1, relative size: min_size
                    s_k_i = self.min_sizes[k]/self.image_size[1]
                    s_k_j = self.min_sizes[k]/self.image_size[0]

                    if len(self.aspect_ratios[0]) == 0:
                        mean += [cx, cy, s_k_i, s_k_j]

                    # Aspect ratio: 1, relative size: sqrt(s_k * s_(k+1))
                    if len(self.max_sizes) == len(self.min_sizes):
                        s_k_prime_i = math.sqrt(s_k_i * (self.max_sizes[k] / self.image_size[1]))
                        s_k_prime_j = math.sqrt(s_k_j * (self.max_sizes[k] / self.image_size[0]))    
                        mean += [cx, cy, s_k_prime_i, s_k_prime_j]

                    # Rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        if len(self.max_sizes) == len(self.min_sizes):
                            mean += [cx, cy, s_k_prime_i/math.sqrt(ar), s_k_prime_j*math.sqrt(ar)]
                        mean += [cx, cy, s_k_i/math.sqrt(ar), s_k_j*math.sqrt(ar)]

        # Convert to torch tensor and reshape
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
