import torch
from torch import nn
#from ssd.data.transforms.target_transform import SSDTargetTransform
#from ssd.data.transforms.transforms import *

class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
   
    
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
     

    
        self.bank0 = torch.nn.Sequential(
                torch.nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[0], kernel_size=3, stride=2, padding=1))
        self.bank1 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[0], 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, output_channels[1], kernel_size=3, stride=2, padding=1))
        self.bank2 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[1], 256, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[2], kernel_size=3, stride=2, padding=1))
        self.bank3 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[2], 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[3], kernel_size=3, stride=2, padding=1))
        self.bank4 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[3], 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, output_channels[4], kernel_size=3, stride=2, padding=1))
        self.bank5 = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels[4], 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, output_channels[5], kernel_size=3, stride=1, padding=0))
    
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        
        out_features = [None] * 6
        out_features[0] = self.bank0(x).cuda()
        out_features[1] = self.bank1(out_features[0]).cuda()
        out_features[2] = self.bank2(out_features[1]).cuda()
        out_features[3] = self.bank3(out_features[2]).cuda()
        out_features[4] = self.bank4(out_features[3]).cuda()
        out_features[5] = self.bank5(out_features[4]).cuda()

        for idx, feature in enumerate(out_features):
            expected_shape = (self.output_channels[idx], self.output_feature_size[idx], self.output_feature_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
