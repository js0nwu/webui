import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score
from torchvision import models
from torch import nn
from pytorch_revgrad import RevGrad
import torch.nn.functional as F
from ui_models_extra import *

class UIScreenEmbedder(pl.LightningModule):
    def __init__(self, hidden_size=256, lr=0.00005, margin_pos=0.2, margin_neg=0.5, lambda_dann=1):
        super(UIScreenEmbedder, self).__init__()
        self.save_hyperparameters()
        
        model = models.resnet18(pretrained=False)
        replace_default_bn_with_in(model)
        model.fc = nn.Linear(model.fc.in_features, hidden_size)
        self.model = model
        
        self.classifier = nn.Sequential(RevGrad(), nn.Linear(model.fc.in_features, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        
    def forward_uda(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        image1 = batch['image1']
        image2 = batch['image2']
        imageuda1 = batch['imageuda1']
        imageuda2 = batch['imageuda2']
        labels = batch['label']
        outs1 = self.model(image1)
        outs2 = self.model(image2)
        batch_size = image1.shape[0]
        delta = outs1 - outs2
        dist = torch.linalg.norm(delta, dim=-1)
        losses = torch.zeros(batch_size, device=self.device)
        losses[labels] = (dist[labels] - self.hparams.margin_pos).clamp(min=0)
        losses[~labels] = (self.hparams.margin_neg - dist[~labels]).clamp(min=0)
        loss_sim = losses.mean()
        if self.hparams.lambda_dann == 0:
            loss = loss_sim
            metrics = {'loss': loss}
            self.log_dict(metrics)
            return loss
        else:
            cls_pred_outs1 = self.forward_uda(image1)
            cls_pred_outs2 = self.forward_uda(image2)
            cls_pred_outsuda1 = self.forward_uda(imageuda1)
            cls_pred_outsuda2 = self.forward_uda(imageuda2)
            cls_pred = torch.cat((cls_pred_outs1, cls_pred_outs2, cls_pred_outsuda1, cls_pred_outsuda2), dim=0).squeeze(-1)
            cls_label = torch.cat((torch.ones(batch_size * 2, device=self.device), torch.zeros(batch_size * 2, device=self.device)), dim=0)
            loss_cls = F.binary_cross_entropy_with_logits(cls_pred, cls_label)
            loss = loss_sim + self.hparams.lambda_dann * loss_cls
            metrics = {'loss': loss, 'loss_sim': loss_sim, 'loss_cls': loss_cls}
            self.log_dict(metrics)
            return loss
    
    def validation_step(self, batch, batch_idx):
        image1 = batch['image1']
        image2 = batch['image2']
        imageuda1 = batch['imageuda1']
        imageuda2 = batch['imageuda2']
        labels = batch['label']
        outs1 = self.model(image1)
        outs2 = self.model(image2)
        batch_size = image1.shape[0]
        delta = outs1 - outs2
        dist = torch.linalg.norm(delta, dim=-1)
        thresh = 0.5 * (self.hparams.margin_pos + self.hparams.margin_neg)
        preds = dist < thresh


        if self.hparams.lambda_dann == 0:
            return preds, labels
        else:
            cls_pred_outs1 = self.forward_uda(image1)
            cls_pred_outs2 = self.forward_uda(image2)
            cls_pred_outsuda1 = self.forward_uda(imageuda1)
            cls_pred_outsuda2 = self.forward_uda(imageuda2)
            cls_pred = torch.cat((cls_pred_outs1, cls_pred_outs2, cls_pred_outsuda1, cls_pred_outsuda2), dim=0).squeeze(-1) > 0
            cls_label = torch.cat((torch.ones(batch_size * 2, device=self.device), torch.zeros(batch_size * 2, device=self.device)), dim=0)
            return preds, labels, cls_pred, cls_label
    
    def validation_epoch_end(self, outputs):
        all_outs = torch.cat([o[0] for o in outputs], dim=0)
        all_labels = torch.cat([o[1] for o in outputs], dim=0)
        score = f1_score(all_labels.detach().cpu().numpy(), all_outs.detach().cpu().numpy())
        
        if self.hparams.lambda_dann == 0:
            metrics = {'f1': score}
            self.log_dict(metrics)
        else:
            all_outs_uda = torch.cat([o[2] for o in outputs], dim=0)
            all_labels_uda = torch.cat([o[3] for o in outputs], dim=0)
            score_uda = f1_score(all_labels_uda.detach().cpu().numpy(), all_outs_uda.detach().cpu().numpy())
            
            metrics = {'f1': score, 'f1_uda': score_uda}
            self.log_dict(metrics)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr)
        return optimizer
