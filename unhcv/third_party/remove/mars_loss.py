from typing import Dict, List

import timeit

import dataclasses

try:
    from CropFormer.api import EntityApi
except ImportError:
    pass
# from efficient_sam.efficient_sam_encoder import ImageEncoderViT, get_abs_pos
from einops import einsum
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from unhcv.common.image import Resize
from unhcv.nn.models.backbone.unet_2d_backbone import zero_module
from unhcv.third_party.remove.segment_anything.modeling.common import LayerNorm2d

try:
    from efficient_sam import build_efficient_sam_vits
    from efficient_sam.efficient_sam import build_efficient_sam
except ImportError:
    pass

from unhcv.common import CfgNode
from unhcv.common.types import ModelOutput
from unhcv.common.utils import find_path
from unhcv.nn.utils import freeze_model, PreNorm, monitor_memory, wrap_no_grad, ReplaceModule

__all__ = ['MARSLoss', 'MARSLossOutput', 'MARSLossConfig']

import torch

@dataclasses.dataclass
class MARSLossOutput(ModelOutput):
    loss: Dict[str, torch.Tensor] = None
    input_image: torch.Tensor = None
    sundries_b_indices: torch.Tensor = None
    pred_mars_masks: torch.Tensor = None
    pred_probs: torch.Tensor = None
    pred_masks_sigmoid: torch.Tensor = None
    pred_masks_filter_sundries: List[torch.Tensor] = None
    pred_masks: torch.Tensor = None
    ios: torch.Tensor = None
    mask_features: torch.Tensor = None

@dataclasses.dataclass
class MARSLossConfig(CfgNode):
    backward_respectively: bool = False
    loss_scale: float = 0.05
    # mask2former
    min_side_size: int = 640
    max_size: int = 1333
    ios_thres: float = 0.9
    score_thres: float = 0.2
    use_mars_mask_loss: bool = True
    use_mars_prob_loss: bool = False
    mask_down_scale: float = 0.25
    # mars_remove_loss
    use_entity_remove: bool = False
    entity_del_sundries: bool = None

class MaskDowner(nn.Module):
    def __init__(self, mask_in_chans=16, activation=nn.GELU, embed_dim=192):
        super().__init__()
        self.mask_downscaling = nn.Sequential(nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
                                              LayerNorm2d(mask_in_chans // 4), activation(),
                                              nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
                                              LayerNorm2d(mask_in_chans), activation())
        self.mask_embed = nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        zero_module(self.mask_embed)

    def forward(self, x):
        x = self.mask_downscaling(x)
        x = self.mask_embed(x)
        return x

class MaskUpper(nn.Module):
    def __init__(self, transformer_dim=256, upscaling_layer_dims=(64, 32), activation=nn.GELU, out_dim=2):
        super().__init__()
        output_dim_after_upscaling = transformer_dim

        self.final_output_upscaling_layers = nn.ModuleList([])
        for idx, layer_dims in enumerate(upscaling_layer_dims):
            self.final_output_upscaling_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        output_dim_after_upscaling,
                        layer_dims,
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.GroupNorm(1, layer_dims)
                    if idx < len(upscaling_layer_dims) - 1
                    else nn.Identity(),
                    activation(),
                )
            )
            output_dim_after_upscaling = layer_dims

        self.output_mask_head = nn.Conv2d(output_dim_after_upscaling, out_dim, kernel_size=1)

    def forward(self, x):
        for up_layer in self.final_output_upscaling_layers:
            x = up_layer(x)
        x = self.output_mask_head(x)
        return x

# class ImageEncoderViTWithMask:
#
#     @staticmethod
#     def forward(self: ImageEncoderViT, x: torch.Tensor, mask_feature: torch.Tensor) -> torch.Tensor:
#         # assert (
#         #     x.shape[2] == self.img_size and x.shape[3] == self.img_size
#         # ), "input image size must match self.img_size"
#         x = self.patch_embed(x)
#         # B C H W -> B H W C
#         x = x.permute(0, 2, 3, 1)
#         x = x + get_abs_pos(
#             self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]]
#         )
#         num_patches = x.shape[1]
#         assert x.shape[2] == num_patches
#         x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])
#         for i_blk, blk in enumerate(self.blocks):
#             if i_blk == self.assert_i_blk:
#                 x = x + mask_feature
#             x = blk(x)
#         x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])
#         x = self.neck(x.permute(0, 3, 1, 2))
#         return x
#
#     @classmethod
#     def init(cls, image_encoder, assert_i_blk=6):
#         ReplaceModule.setattr_function(image_encoder, "forward", cls.forward)
#         image_encoder.assert_i_blk = assert_i_blk

class MaskFormerLoss:

    @staticmethod
    def forward(self, image):
        features = self.backbone(image)
        outputs = self.sem_seg_head(features)
        return outputs

    @classmethod
    def init(cls, model):
        ReplaceModule.setattr_function(model, "forward", cls.forward)
        return model

class MARSLoss(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config = MARSLossConfig.from_other(config)
        config_file = find_path(
            "code/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/mask2former_swin_tiny_3x.yaml")
        entity_api = EntityApi(config_file=config_file)
        entity_model = entity_api.demo.predictor.model
        self.entity_model = MaskFormerLoss.init(entity_model)
        freeze_model(self.entity_model)
        self.cross_loss = CrossEntropyLoss()
        self.pre_norm = PreNorm(pixel_mean=IMAGENET_DEFAULT_MEAN, pixel_std=IMAGENET_DEFAULT_STD)
        '''
        efficient_sam = build_efficient_sam(encoder_patch_embed_dim=384, encoder_num_heads=6,
                                            checkpoint=find_path("model/EfficientSAM/efficient_sam_vits.pt"))
        self.image_encoder = efficient_sam.image_encoder
        ImageEncoderViTWithMask.init(image_encoder=self.image_encoder)
        self.mask_downer = MaskDowner(embed_dim=384)
        self.mask_upper = MaskUpper()
        '''

    def _forward_one(self, predict_images, inpainting_masks_resized, *, only_mars_mask=False, ios_thres):
        batch_size = predict_images.size(0)
        predict = self.entity_model(predict_images)
        pred_masks:torch.Tensor = predict['pred_masks']
        pred_masks_sigmoid = pred_masks.sigmoid()
        pred_probs = predict['pred_logits'].softmax(-1)[..., 0]
        pred_logits = predict['pred_logits']
        with torch.no_grad():
            ios = einsum(pred_masks_sigmoid, inpainting_masks_resized, 'n c h w, n k h w -> n c')
            ios = ios / einsum(pred_masks_sigmoid, 'n c h w -> n c').clamp_(min=1e-4)
            if ios_thres is None:
                ios_thres = self.config.ios_thres
        sundries_indices = sundries_indices_ios = ios > ios_thres # type: ignore
        sundries_indices: torch.Tensor = sundries_indices & (pred_probs > self.config.score_thres)
        if self.config.use_entity_remove:
            pred_masks_filter_sundries = []
            with torch.no_grad():
                for i_pred, (pred_mask, pred_prob) in enumerate(zip(pred_masks.detach(), pred_probs.detach())):
                    score_valid = pred_prob > self.config.score_thres
                    if self.config.entity_del_sundries:
                        pred_sundries_mask = pred_mask[score_valid & sundries_indices_ios[i_pred]]
                    valid = score_valid & ~sundries_indices_ios[i_pred]  # type: ignore
                    pred_mask = pred_mask[valid]
                    if self.config.entity_del_sundries:
                        if pred_sundries_mask.size(0) > 0:
                            # TODO: set score to parameter
                            pred_sundries_mask = pred_sundries_mask.max(0)[0].sigmoid() > 0.2
                            pred_mask[:, pred_sundries_mask] = -torch.inf

                    pred_masks_filter_sundries.append(pred_mask)
        else:
            pred_masks_filter_sundries = None
        if only_mars_mask:
            pred_mars_masks = []
            for i_mask, sundries_index in enumerate(sundries_indices):
                if sundries_index.any():
                    pred_mars_masks.append(pred_masks[i_mask, sundries_index].max(0)[0])
                else:
                    pred_mars_masks.append(pred_masks.new_full(pred_masks.shape[2:], fill_value=-torch.inf))
            pred_mars_masks = torch.stack(pred_mars_masks, dim=0)
            return MARSLossOutput(pred_mars_masks=pred_mars_masks, pred_masks_sigmoid=pred_masks_sigmoid.detach(), pred_probs=pred_probs.detach(), ios=ios, pred_masks=pred_masks, pred_masks_filter_sundries=pred_masks_filter_sundries)
        sundries_b_indices = torch.where(sundries_indices)[0]
        loss_dict = {}
        if len(sundries_b_indices) == 0:
            if self.config.use_mars_mask_loss:
                mars_mask_loss = pred_masks.new_tensor(0)
                loss_dict['mars_mask_loss'] = mars_mask_loss
            if self.config.use_mars_prob_loss:
                mars_prob_loss = pred_masks.new_tensor(0)
                loss_dict['mars_prob_loss'] = mars_prob_loss
            pred_mars_masks = None
        else:
            pred_mars_masks = pred_masks[sundries_indices]
            if self.config.use_mars_mask_loss:
                inpainting_masks_mar = inpainting_masks_resized[sundries_b_indices, 0]
                mars_mask_loss = F.binary_cross_entropy_with_logits(pred_mars_masks.float(), torch.zeros_like(pred_mars_masks, dtype=torch.float), reduction='none')
                mars_mask_loss = mars_mask_loss * inpainting_masks_mar
                # mars_mask_loss = einsum(mars_mask_loss, 'n h w -> n') / einsum(inpainting_masks_mar, 'n h w -> n').clamp_(min=1e-4)
                mars_mask_loss = einsum(mars_mask_loss, 'n h w -> n') / (mars_mask_loss.size(1) * mars_mask_loss.size(2))
                loss_dict['mars_mask_loss'] = mars_mask_loss.sum() * (self.config.loss_scale / batch_size)
            if self.config.use_mars_prob_loss:
                pred_mars_logits = pred_logits[sundries_indices]
                mars_prob_loss = F.cross_entropy(pred_mars_logits.float(), pred_mars_logits.new_full(pred_mars_logits.shape[0:1], dtype=torch.long, fill_value=pred_mars_logits.shape[-1] - 1), reduction='none')
                pred_mars_masks_detach = (pred_mars_masks > 0).to(pred_mars_masks).detach().flatten(1)
                pred_mars_masks_detach_ratio = pred_mars_masks_detach.sum(-1); pred_mars_masks_detach_ratio /= (pred_mars_masks.size(1) * pred_mars_masks.size(2))
                mars_prob_loss = mars_prob_loss * pred_mars_masks_detach_ratio
                loss_dict['mars_prob_loss'] = mars_prob_loss.sum() * (self.config.loss_scale / batch_size)

        return MARSLossOutput(loss=loss_dict, sundries_b_indices=sundries_b_indices, pred_mars_masks=pred_mars_masks,
                              input_image=predict_images, pred_masks_sigmoid=pred_masks_sigmoid.detach(), pred_masks=pred_masks,
                              pred_probs=pred_probs.detach(), pred_masks_filter_sundries=pred_masks_filter_sundries,
                              mask_features=predict["mask_features"]) # type: ignore

    def forward(self, predict_images, inpainting_masks, entity_masks=None, only_mars_mask=False, ios_thres=None):
        """
        predict_images: n c h w, 0-1
        inpainting_masks: n 1 h w, 0-1
        """
        tgt_size = Resize.cal_target_size(predict_images.shape[2:], max_size=self.config.max_size, min_side_size=self.config.min_side_size, size_divisibility=32)
        predict_images = F.interpolate(predict_images, tgt_size, mode="bilinear", align_corners=False)
        predict_images = self.pre_norm(predict_images)
        inpainting_masks_resized = F.interpolate(inpainting_masks, size=(int(tgt_size[0] * self.config.mask_down_scale),
                                                                         int(tgt_size[1] * self.config.mask_down_scale)),
                                                 mode="nearest-exact")
        batch_size = predict_images.shape[0]
        if self.config.backward_respectively:
            raise NotImplementedError
            # for i in range(batch_size):
            #     predict_images_ = predict_images[i:i+1]; inpainting_masks_resized_ = inpainting_masks_resized[i:i+1]
            #     mars_output = self._forward_one(predict_images_, inpainting_masks_resized_)
        else:
            # for save memory
            mars_output = self._forward_one(predict_images, inpainting_masks_resized, only_mars_mask=only_mars_mask, ios_thres=ios_thres)

        return mars_output

    @wrap_no_grad
    def predict_mars_masks(self, predict_images, inpainting_masks, *, entity_masks=None, ios_thres, norm_01=True):
        if norm_01:
            predict_images = predict_images * 0.5 + 0.5
        mars_output = self.forward(predict_images, inpainting_masks, entity_masks, only_mars_mask=True, ios_thres=ios_thres)
        return mars_output

if __name__ == '__main__':
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import torch
    from PIL import Image
    import requests

    dtype = torch.bfloat16
    batch_size = 8
    mars_loss = MARSLoss().to(dtype).cuda()
    predict_images = torch.randn([1, 3, 600, 600]).cuda().to(dtype)
    # predict_images = to_tensor(obj_load('/home/yixing/dataset/layout_inpaint/demo_0731/image/0003.jpg'))[None].cuda().to(dtype)
    inpainting_masks = predict_images.new_ones((predict_images.size(0), 1, *predict_images.shape[2:])).to(dtype)
    predict_images = predict_images.repeat(batch_size, 1, 1, 1)
    inpainting_masks = inpainting_masks.repeat(batch_size, 1, 1, 1)
    predict_images.requires_grad_(True)
    loss = mars_loss(predict_images, inpainting_masks)
    print(monitor_memory())
    del loss

    def test():
        loss = mars_loss(predict_images, inpainting_masks)
        del loss
    def test1():
        with torch.no_grad():
            loss = mars_loss(predict_images, inpainting_masks)
    print(timeit.timeit(test, number=10))
    print(timeit.timeit(test, number=10))
    print(timeit.timeit(test1, number=10))
    monitor_memory()


    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained('/home/yixing/model/Depth-Anything-V2-Small-hf')
    model = AutoModelForDepthEstimation.from_pretrained('/home/yixing/model/Depth-Anything-V2-Small-hf')

    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # interpolate to original size and visualize the prediction
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth = depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8"))