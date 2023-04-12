import torch
from torch import nn
from torchvision.ops import StochasticDepth
from torchvision.models.resnet import BasicBlock, Bottleneck

class StochasticBasicBlock(nn.Module):
    def __init__(self, m, stochastic_depth_p=0.2, stochastic_depth_mode="row"):
        super(StochasticBasicBlock, self).__init__()
        self.m = m
        self.sd = StochasticDepth(stochastic_depth_p, mode=stochastic_depth_mode)
    
    def forward(self, x):
        identity = x

        out = self.m.conv1(x)
        out = self.m.bn1(out)
        out = self.m.relu(out)

        out = self.m.conv2(out)
        out = self.m.bn2(out)

        out = self.sd(out)
        
        if self.m.downsample is not None:
            identity = self.m.downsample(x)

        out += identity
        out = self.m.relu(out)

        return out

class StochasticBottleneck(nn.Module):
    def __init__(self, m, stochastic_depth_p=0.2, stochastic_depth_mode="row"):
        super(StochasticBottleneck, self).__init__()
        self.m = m
        self.sd = StochasticDepth(stochastic_depth_p, mode=stochastic_depth_mode)
    
    def forward(self, x):
        identity = x

        out = self.m.conv1(x)
        out = self.m.bn1(out)
        out = self.m.relu(out)

        out = self.m.conv2(out)
        out = self.m.bn2(out)
        out = self.m.relu(out)

        out = self.m.conv3(out)
        out = self.m.bn3(out)
        
        out = self.sd(out)

        if self.m.downsample is not None:
            identity = self.m.downsample(x)

        out += identity
        out = self.m.relu(out)

        return out

class CustomNormAndDropout(nn.Module):
    def __init__(self, num_features, dropout):
        super(CustomNormAndDropout, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(x)
        return x

def replace_default_bn_with_custom(model, dropout=0.0):
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, CustomNormAndDropout(child.num_features, dropout))
        else:
            replace_default_bn_with_custom(child, dropout)

def replace_default_bn_with_in(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, child_name, nn.InstanceNorm2d(child.num_features))
        else:
            replace_default_bn_with_in(child)

def replace_res_blocks_with_stochastic(model, stochastic_depth_p=0.2, stochastic_depth_mode="row"):
    all_blocks = []
    def get_blocks(model, blocks):
        for child_name, child in model.named_children():
            if isinstance(child, BasicBlock):
                # setattr(model, child_name, StochasticBasicBlock(child, stochastic_depth_p, stochastic_depth_mode))
                blocks.append((child_name, StochasticBasicBlock))
            elif isinstance(child, Bottleneck):
                # setattr(model, child_name, StochasticBottleneck(child, stochastic_depth_p, stochastic_depth_mode))
                blocks.append((child_name, StochasticBottleneck))
            else:
                get_blocks(child, blocks)
    get_blocks(model, all_blocks)
    p_alphas = torch.linspace(0, 1, len(all_blocks)) * stochastic_depth_p
    for bi in range(len(all_blocks)):
        setattr(model, all_blocks[bi][0], all_blocks[bi][1](p_alphas[bi], stochastic_depth_mode))