import os
from dataclasses import dataclass
from enum import Enum #, StrEnum
from typing import Tuple, Any, List, Dict

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import configurable, get_cfg, CfgNode
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone as build_backbone_, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry

from mask2former import MaskFormer, add_maskformer2_config
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher

from unhcv.common.image import mask_iou
from unhcv.common.utils import find_path, obj_load
from unhcv.nn.models.backbone import Unet2DBackbone, BackboneOutput
from unhcv.nn.utils import load_checkpoint

# from unhcv.projects.segmentation.unet_2d_backbone import Unet2DBackbone, LDMInput, IndexConfig, FPNBackbone


CRITERION_REGISTRY = Registry("TRANSFORMER_MODULE")
CRITERION_REGISTRY.__doc__ = """
Registry for criterion.
"""
CRITERION_REGISTRY.register(SetCriterion)
CRITERION_REGISTRY.register(HungarianMatcher)


@dataclass
class CustomExtraMask2FormerConfig:
    backbone_config: Any = None
    backbone_pretrain_model: str = None


@dataclass
class Mask2FormerInput:
    labels: List[torch.Tensor] = None
    masks: List[torch.Tensor] = None
    images: torch.Tensor = None


class BackboneName(str, Enum):
    Unet2DBackbone = "Unet2D"


class Unet2DBackbone(Unet2DBackbone, Backbone):
    @property
    def size_divisibility(self) -> int:
        return 0

    @property
    def padding_constraints(self) -> Dict[str, int]:
        return {}

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        channels = self.channels
        output_shape = {}
        for i in range(len(channels)):
            output_shape[f"res{2+i}"] = ShapeSpec(channels=channels[i], stride=self.strides[i])
        return output_shape

    def forward(self, *args, **kwargs) -> Dict:
        features = super().forward(*args, **kwargs).backbone_features
        out = {}
        for i in range(len(features)):
            out[f"res{2+i}"] = features[i]
        return out


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    if backbone_name == BackboneName.Unet2DBackbone:
        pretrained_model = find_path(cfg.MODEL.BACKBONE.PRETRAINED_MODEL)
        config = obj_load(os.path.join(pretrained_model, "config.json"))
        config.update(dict(out_channels=None, in_channels=4))
        pretrained_state_dict = os.path.join(pretrained_model, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(pretrained_state_dict):
            pretrained_state_dict = os.path.join(pretrained_model, "diffusion_pytorch_model.bin")
        backbone = Unet2DBackbone.from_config(config)
        load_checkpoint(backbone, pretrained_state_dict, log_missing_keys=True, mismatch_shape=True)
    else:
        backbone = build_backbone_(cfg, input_shape)
    return backbone


@META_ARCH_REGISTRY.register()
class CustomMaskFormer(MaskFormer):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        match_name = cfg.MODEL.MASK_FORMER.get("MATCH_NAME", "HungarianMatcher")
        matcher = CRITERION_REGISTRY.get(match_name)(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion_name = cfg.MODEL.MASK_FORMER.get("CRITERION_NAME", "SetCriterion")
        criterion = CRITERION_REGISTRY.get(criterion_name)(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    def calculate_loss(self, batched_inputs, outputs):
        # mask classification target
        targets = [dict(masks=masks, labels=labels) for masks, labels in zip(batched_inputs.mask_labels, batched_inputs.class_labels)]

        # bipartite matching-based loss
        losses = self.criterion(outputs, targets)
        return losses

    def inference_forward(self, batched_inputs: Mask2FormerInput, outputs):
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result in zip(
            mask_cls_results, mask_pred_results
        ):
            height = image_size[0]
            width = image_size[1]
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # instance segmentation inference
            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["instances"] = instance_r
        return processed_results

    def forward(self, batched_inputs: Mask2FormerInput):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        # images = [x["image"].to(self.device) for x in batched_inputs]
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(batched_inputs.images)
        outputs = self.sem_seg_head(features)

        if self.training:
            losses = self.calculate_loss(batched_inputs, outputs)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            processed_results = self.inference_forward(batched_inputs, outputs)
            return processed_results


def add_custom_config(cfg):
    """
    Add config for CUSTOM MASK_FORMER.
    """
    cfg.MODEL.MASK_FORMER.MATCH_NAME = "HungarianMatcher"
    cfg.MODEL.MASK_FORMER.CRITERION_NAME = "SetCriterion"
    cfg.MODEL.BACKBONE.PRETRAINED_MODEL = None
    cfg.MODEL.NORM_MEAN = [0.485, 0.456, 0.406] # [123.675, 116.280, 103.530]
    cfg.MODEL.NORM_STD = [0.229, 0.224, 0.225] # [58.395, 57.120, 57.375]


def build_mask2former_model(config_file, extra_config=dict(), add_config_func=None, only_return_config=False):
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_custom_config(cfg)
    if add_config_func is not None:
        add_config_func(cfg)
    if config_file is not None:
        cfg.merge_from_file(config_file)
    cfg.extra_config = CfgNode(extra_config)
    if only_return_config:
        return cfg
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    # model = CustomMaskFormer(cfg)
    return model, cfg
