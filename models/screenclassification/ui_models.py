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
        if arch == "resnet50" or arch == "resnet50_conv":
            model = models.resnet50(pretrained=False)
            replace_default_bn_with_custom(model, dropout=dropout_block)
            replace_res_blocks_with_stochastic(model, stochastic_depth_p=stochastic_depth_p)

            model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
            self.model = model

            # self.conv_cls = nn.Conv2d(2048, 1, 3 stride=1, padding=1)
            self.conv_cls = nn.Sequential(
                nn.InstanceNorm2d(2048),
                nn.Dropout2d(dropout),
                nn.Conv2d(2048, num_classes, 3, stride=1, padding=1)
            )
        elif arch == "vgg16":
            model = models.vgg16_bn(pretrained=False, dropout=dropout)
            replace_default_bn_with_custom(model, dropout=dropout_block)
            model.classifier[-1] = nn.Linear(4096, num_classes)
            self.model = model

    def forward(self, image):
        if self.hparams.arch == "resnet50" or self.hparams.arch == "vgg16":
            return self.model(image)
        elif self.hparams.arch == "resnet50_conv":
            x = self.model.conv1(image)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.conv_cls(x)
            
            batch_size = x.shape[0]
            res = x.view(batch_size, self.hparams.num_classes, -1).mean(dim=-1)
            return res
        
    def training_step(self, batch, batch_idx):
        # image, labels = batch
        image = batch['image']
        labels = batch['label']
        # out = self.model(image)
        # out = self.model(torch.stack(image, dim=0))
        outs = [self.forward(image[i].unsqueeze(0)) for i in range(len(image))]
        out = torch.cat(outs, dim=0)
        # loss = self.loss_fn(out,labels)
        if len(labels.shape) == 2:
            if self.hparams.soft_labels:
                loss = F.cross_entropy(out, labels.float())
            else:
                loss = F.binary_cross_entropy_with_logits(out, labels)
        else:
            loss = F.cross_entropy(out, labels)
        
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        labels = batch['label']
        # out = self.model(image)
        # out = self.model(torch.stack(image, dim=0))
        outs = [self.forward(image[i].unsqueeze(0)) for i in range(len(image))]
        out = torch.cat(outs, dim=0)
        if len(labels.shape) == 2:
            return out, labels
        else:
            _, inds = out.max(dim=-1)
            return inds, labels

    def validation_epoch_end(self, outputs):
        all_outs = torch.cat([o[0] for o in outputs], dim=0)
        all_labels = torch.cat([o[1] for o in outputs], dim=0)
        if len(all_labels.shape) == 2:
            bce_score = F.binary_cross_entropy_with_logits(all_outs, all_labels)
            score_dict = {'bce': bce_score}
            print(score_dict)
            self.log_dict(score_dict)
        else:
            all_outs = all_outs.detach().cpu().long().numpy()
            all_labels = all_labels.detach().cpu().long().numpy()
            macro_score = f1_score(all_labels, all_outs, average="macro")
            micro_score = f1_score(all_labels, all_outs, average="micro")
            weighted_score = f1_score(all_labels, all_outs, average="weighted")
            score_dict = {'f1_macro': macro_score, 'f1_micro': micro_score, 'f1_weighted': weighted_score}
            print(score_dict)
            self.log_dict(score_dict)
            
    def test_step(self, batch, batch_idx):
        image = batch['image']
        labels = batch['label']
        # out = self.model(image)
        # out = self.model(torch.stack(image, dim=0))
        outs = [self.forward(image[i].unsqueeze(0)) for i in range(len(image))]
        out = torch.cat(outs, dim=0)
        if len(labels.shape) == 2:
            return out, labels
        else:
            _, inds = out.max(dim=-1)
            return inds, labels

    def test_epoch_end(self, outputs):
        all_outs = torch.cat([o[0] for o in outputs], dim=0)
        all_labels = torch.cat([o[1] for o in outputs], dim=0)
        if len(all_labels.shape) == 2:
            bce_score = F.binary_cross_entropy_with_logits(all_outs, all_labels)
            score_dict = {'bce': bce_score}
            print(score_dict)
            self.log_dict(score_dict)
        else:
            all_outs = all_outs.detach().cpu().long().numpy()
            all_labels = all_labels.detach().cpu().long().numpy()
            macro_score = f1_score(all_labels, all_outs, average="macro")
            micro_score = f1_score(all_labels, all_outs, average="micro")
            weighted_score = f1_score(all_labels, all_outs, average="weighted")
            score_dict = {'f1_macro': macro_score, 'f1_micro': micro_score, 'f1_weighted': weighted_score}
            print(score_dict)
            return score_dict

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=0)
        # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.002, max_lr=0.01)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": lr_scheduler
        # }
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr)
        # optimizer = torch.optim.RMSprop([p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr, momentum=0.9)
        # lr_scheduler = ReduceLROnPlateau(optimizer, 'max')
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": lr_scheduler,
        #     "monitor": "f1_micro"
        # }
        return optimizer

class UIScreenSegmenter(pl.LightningModule):
    def __init__(self, num_classes=20):
        super(UIScreenSegmenter, self).__init__()
        self.save_hyperparameters()
        model = models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

        self.decoder = nn.Sequential(
            nn.InstanceNorm2d(2048),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2048, 1024, 3, stride=1, padding=1),
            nn.InstanceNorm2d(1024),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 3, stride=1, padding=1)
        )
        
    def encode(self, img):
        # z = self.model(image)
        # out = self.l1(z)
        # out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # img = self.conv_blocks(out)
        # return img
        x = self.model.conv1(img)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x

    def decode(self, x, out_size):
        x = self.decoder(x)
        return F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        return self.decode(self.encode(x), (x.shape[-2], x.shape[-1]))

    def training_step(self, batch, batch_idx):
        # image, labels = batch
        image = batch['image']
        segmentation = batch['segmentation']
        outs = [self.forward(image[i].unsqueeze(0)) for i in range(len(image))]

        losses = [F.binary_cross_entropy_with_logits(outs[i], segmentation[i].unsqueeze(0)) for i in range(len(outs))]
        loss = torch.stack(losses).mean()
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        segmentation = batch['segmentation']
        outs = [self.forward(image[i].unsqueeze(0)) for i in range(len(image))]

        losses = [F.binary_cross_entropy_with_logits(outs[i], segmentation[i].unsqueeze(0)) for i in range(len(outs))]
        loss = torch.stack(losses).mean()
        return loss

    def validation_epoch_end(self, outputs):
        bce_score = torch.stack(outputs).mean()
        score_dict = {'bce': bce_score}
        print(score_dict)
        self.log_dict(score_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }
