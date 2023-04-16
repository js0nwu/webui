from typing import Any, Callable, Dict, List, Tuple, Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from torchvision.models.detection.fcos import FCOSHead

from torchvision.ops import sigmoid_focal_loss, generalized_box_iou_loss

class FCOSMultiHead(FCOSHead):
    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:
        cls_logits = head_outputs["cls_logits"]  # [N, HWA, C]
        bbox_regression = head_outputs["bbox_regression"]  # [N, HWA, 4]
        bbox_ctrness = head_outputs["bbox_ctrness"]  # [N, HWA, 1]

        target_lens = [len(t["labels"].shape) for t in targets]
        multi_head_mode = any([tl == 2 for tl in target_lens])
        all_gt_classes_targets = []
        all_gt_boxes_targets = []
        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            if len(targets_per_image["labels"]) == 0:
                if multi_head_mode:
                    gt_classes_targets = targets_per_image["labels"].new_zeros((len(matched_idxs_per_image), cls_logits.shape[2]))
                else:
                    gt_classes_targets = targets_per_image["labels"].new_zeros((len(matched_idxs_per_image),))
                gt_boxes_targets = targets_per_image["boxes"].new_zeros((len(matched_idxs_per_image), 4))
            else:
                gt_classes_targets = targets_per_image["labels"][matched_idxs_per_image.clip(min=0)]
                gt_boxes_targets = targets_per_image["boxes"][matched_idxs_per_image.clip(min=0)]
            gt_classes_targets[matched_idxs_per_image < 0] = -1  # backgroud
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_boxes_targets)

        all_gt_classes_targets = torch.stack(all_gt_classes_targets)
        # compute foregroud
        if len(all_gt_classes_targets.shape) == 3:
            foregroud_mask = all_gt_classes_targets.min(dim=-1)[0] >= 0
        else:
            foregroud_mask = all_gt_classes_targets >= 0
        num_foreground = foregroud_mask.sum().item()

        # classification loss
        if len(all_gt_classes_targets.shape) == 3:
            gt_classes_targets = all_gt_classes_targets.float()
            loss_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets.clamp(min=0, max=1), reduction="sum") # try mean so that it is lower in comparison to the position regressor       
        else:
            gt_classes_targets = torch.zeros_like(cls_logits)
            gt_classes_targets[foregroud_mask, all_gt_classes_targets[foregroud_mask]] = 1.0
            loss_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="sum")

        # regression loss: GIoU loss
        # TODO: vectorize this instead of using a for loop
#        pred_boxes = [
#            self.box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
#            for anchors_per_image, bbox_regression_per_image in zip(anchors, bbox_regression)
#        ]
        pred_boxes = self.box_coder.decode(bbox_regression, torch.stack(anchors))
        
        # amp issue: pred_boxes need to convert float
        
        loss_bbox_reg = F.mse_loss(pred_boxes[foregroud_mask].float(), torch.stack(all_gt_boxes_targets)[foregroud_mask], reduction="mean")

        # ctrness loss
#        bbox_reg_targets = [
#            self.box_coder.encode_single(anchors_per_image, boxes_targets_per_image)
#            for anchors_per_image, boxes_targets_per_image in zip(anchors, all_gt_boxes_targets)
#        ]
#        bbox_reg_targets = torch.stack(bbox_reg_targets, dim=0)

        bbox_reg_targets = self.box_coder.encode(torch.stack(anchors), torch.stack(all_gt_boxes_targets))
        
        if len(bbox_reg_targets) == 0:
            gt_ctrness_targets = bbox_reg_targets.new_zeros(bbox_reg_targets.size()[:-1])
        else:
            left_right = bbox_reg_targets[:, :, [0, 2]]
            top_bottom = bbox_reg_targets[:, :, [1, 3]]
            gt_ctrness_targets = torch.sqrt(
                (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
                * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
            )
        pred_centerness = bbox_ctrness.squeeze(dim=2)
        loss_bbox_ctrness = nn.functional.binary_cross_entropy_with_logits(
            pred_centerness[foregroud_mask], gt_ctrness_targets[foregroud_mask], reduction="sum"
        )

        retval = {
            "classification": loss_cls / max(1, num_foreground),
            "bbox_regression": loss_bbox_reg / max(1, num_foreground),
            "bbox_ctrness": loss_bbox_ctrness / max(1, num_foreground),
        }
        # print(retval)
        return retval
