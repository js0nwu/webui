import torch
from torch import nn

def replace_default_bn_with_in(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, nn.InstanceNorm2d(child.num_features))
        else:
            replace_default_bn_with_in(child)
