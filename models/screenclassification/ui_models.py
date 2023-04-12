import torchvision.models as models

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torchvision
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ui_models_extra import replace_default_bn_with_custom, replace_res_blocks_with_stochastic, replace_default_bn_with_in

class UIScreenClassifier(pl.LightningModule): # support resnet50 or vgg16 config used in enrico paper
    def __init__(self, num_classes=20, dropout_block=0.0, dropout=0.2, lr=0.00005, soft_labels=True, stochastic_depth_p=0.2, arch="resnet50"):

        super(UIScreenClassifier, self).__init__()
        self.save_hyperparameters()
        if arch == "resnet50":
            model = models.resnet50(pretrained=False)
            replace_default_bn_with_custom(model, dropout=dropout_block)
            replace_res_blocks_with_stochastic(model, stochastic_depth_p=stochastic_depth_p)

            model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
            self.model = model
        elif arch == "vgg16":
            model = models.vgg16_bn(pretrained=False, dropout=dropout)
            replace_default_bn_with_custom(model, dropout=dropout_block)
            model.classifier[-1] = nn.Linear(4096, num_classes)
            self.model = model

    def forward(self, image):
        return self.model(image)
        
    def training_step(self, batch, batch_idx):
        image = batch['image']
        labels = batch['label']
        outs = [self.forward(image[i].unsqueeze(0)) for i in range(len(image))]
        out = torch.cat(outs, dim=0)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        labels = batch['label']
        outs = [self.forward(image[i].unsqueeze(0)) for i in range(len(image))]
        out = torch.cat(outs, dim=0)
        _, inds = out.max(dim=-1)
        return inds, labels

    def validation_epoch_end(self, outputs):
        all_outs = torch.cat([o[0] for o in outputs], dim=0)
        all_labels = torch.cat([o[1] for o in outputs], dim=0)
        all_outs = all_outs.detach().cpu().long().numpy()
        all_labels = all_labels.detach().cpu().long().numpy()
        macro_score = f1_score(all_labels, all_outs, average="macro")
        micro_score = f1_score(all_labels, all_outs, average="micro")
        weighted_score = f1_score(all_labels, all_outs, average="weighted")
        score_dict = {'f1_macro': macro_score, 'f1_micro': micro_score, 'f1_weighted': weighted_score}
        self.log_dict(score_dict)
            
    def test_step(self, batch, batch_idx):
        image = batch['image']
        labels = batch['label']
        outs = [self.forward(image[i].unsqueeze(0)) for i in range(len(image))]
        out = torch.cat(outs, dim=0)
        _, inds = out.max(dim=-1)
        return inds, labels

    def test_epoch_end(self, outputs):
        all_outs = torch.cat([o[0] for o in outputs], dim=0)
        all_labels = torch.cat([o[1] for o in outputs], dim=0)
        all_outs = all_outs.detach().cpu().long().numpy()
        all_labels = all_labels.detach().cpu().long().numpy()
        macro_score = f1_score(all_labels, all_outs, average="macro")
        micro_score = f1_score(all_labels, all_outs, average="micro")
        weighted_score = f1_score(all_labels, all_outs, average="weighted")
        score_dict = {'f1_macro': macro_score, 'f1_micro': micro_score, 'f1_weighted': weighted_score}
        return score_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr)
        return optimizer