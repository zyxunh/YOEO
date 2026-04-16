import timeit

import dataclasses

from diffusers.models.attention import BasicTransformerBlock
from torch import nn
from torchvision.transforms.functional import to_tensor
from typing import Literal

import torch
import torch.nn.functional as F

from unhcv.nn.functional import one_hot, fix_grad_forward
# from unhcv.third_party.dino import DinoV3

try:
    from efficient_sam import build_efficient_sam_vits
    from efficient_sam.efficient_sam import build_efficient_sam
except ImportError:
    pass
from einops import rearrange, einsum
from transformers import Dinov2Model
from transformers.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from unhcv.common import CfgNode
from unhcv.common.image import masks2bboxes, ratio_length2hw, visual_tensor
from unhcv.common.image.geometric import expand_box
from unhcv.common.types import ModelOutput, DataDict, PathStr
from unhcv.common.utils import find_path
from unhcv.nn.utils import freeze_model, PreNorm, monitor_memory, wrap_no_grad
from unhcv.third_party.remove.segment_anything import sam_model_registry, SamPredictor


__all__ = ['RemoveLossOutput', 'RemoveLossConfig', 'RemoveLoss']

@dataclasses.dataclass
class RemoveLossConfig(CfgNode):
    image_encoder_name: Literal["sam", "dinov2", "efficient_sam", "mars"] = "dinov2"
    image_encoder_pretrained_path: PathStr = "model/dinov2-base"
    image_size: int = 224
    pad_on: bool = False
    use_crop: bool = True
    #
    inpaint_resize_mode: Literal["mean_pool", "nearest"] = "mean_pool"
    #
    use_mean_loss: bool = False
    backward_respectively: bool = False
    # for entity similar
    use_entity_similar: bool = False
    use_predict_entity: bool = False
    use_global_similar: bool = True
    entity_del_sundries: bool = False
    use_teacher_entity: bool = False
    use_teacher_entity_on_pair_data: bool = False
    only_teacher_entity: bool = False
    # for predict entity similar
    predict_prob_thres: float = 1.
    # gan
    use_gan: bool = False
    num_transformer_blocks: int = 2

    def __post_init__(self):
        pass


@dataclasses.dataclass
class RemoveLossOutput(ModelOutput):
    loss: torch.Tensor = None
    image_features: torch.Tensor = None
    predict_image_rect: torch.Tensor = None
    predict_image_rect_grad: torch.Tensor = None
    cluster_entities: torch.Tensor = None
    inpainting_mask_rect: torch.Tensor = None
    entity_mask_rect: torch.Tensor = None

class RemoveLoss(torch.nn.Module):
    def __init__(self, config=None):
        self.config = config = RemoveLossConfig.from_other(config)
        super(RemoveLoss, self).__init__()
        if config.image_encoder_name == "sam":
            sam = sam_model_registry["vit_h"](checkpoint=find_path("model/sam/sam_vit_h_4b8939.pth")).cuda()
            freeze_model(sam)
            self.predictor = SamPredictor(sam)
        elif config.image_encoder_name == "efficient_sam":
            efficient_sam = build_efficient_sam(encoder_patch_embed_dim=384, encoder_num_heads=6, checkpoint=find_path("model/EfficientSAM/efficient_sam_vits.pt"),).eval()
            encoder_dim = 256
            self.image_encoder = efficient_sam.image_encoder
        elif config.image_encoder_name == "dinov2":
            self.image_encoder = Dinov2Model.from_pretrained(config.image_encoder_pretrained_path)
        elif config.image_encoder_name == "dinov3":
            self.image_encoder = DinoV3()
        if config.image_encoder_name in {"dinov2", "efficient_sam", "dinov3"}:
            freeze_model(self.image_encoder)
        self.pre_norm = PreNorm(pixel_mean=IMAGENET_DEFAULT_MEAN, pixel_std=IMAGENET_DEFAULT_STD)
        if config.use_gan:
            gan_input_size = (64, 64)
            from unhcv.third_party.distill.gan_model import GanTransformers
            self.gan_transformers = GanTransformers(in_channels=encoder_dim, num_transformer_blocks=config.num_transformer_blocks, heads=8, token_num=gan_input_size[0] * gan_input_size[1] + 2)

        self.expand_ratio = 0.2

    @property
    def image_mean_std(self):
        return self.pre_norm.pixel_mean.view(-1), self.pre_norm.pixel_std.view(-1)

    def feed_models(self, models):
        self.gan_transformers = models

    def resize_mask(self, mask, hw):
        if self.config.inpaint_resize_mode == "mean_pool":
            mask = F.adaptive_avg_pool2d(mask, hw)
        elif self.config.inpaint_resize_mode == "nearest":
            mask = F.interpolate(mask, size=hw, mode="nearest-exact")
        return mask
    def cal_masked_mean_feature(self, features, mask, resize=True):
        features_hw = features.shape[2:]
        if resize:
            mask = self.resize_mask(mask, features_hw)
        if self.config.inpaint_resize_mode == "mean_pool":
            masked_mean_feature = einsum(features, mask, 'n c h w, n k h w -> n c k')
            mask_sum = einsum(mask, 'n k h w -> n k')
            masked_mean_feature = masked_mean_feature / mask_sum[:, None].clamp(min=1e-4)
            # outpainting_mask_rect = 1 - mask
            # feature_out_inpaint = einsum(features, outpainting_mask_rect, 'n c h w, n k h w -> n c')
            # feature_out_inpaint = feature_out_inpaint / einsum(outpainting_mask_rect, 'n k h w -> n k').clamp(min=1e-4)
        elif self.config.inpaint_resize_mode == "nearest":
            pass
        return masked_mean_feature, mask_sum

    def get_loss(self, predict_image_rect, inpainting_mask_rect, entity_mask_rect, pred_masks_filter_sundries, *, labels=None, features=None, i_image=None, mars_output_teacher, data_tag):
        remove_loss = {}; cluster_entity = None
        if self.config.image_encoder_name == "sam":
            features = self.predictor.model.image_encoder(predict_image_rect)
            remove_loss = None
        elif self.config.image_encoder_name in {"dinov2", "efficient_sam", "dinov3", "mars"}:
            if self.config.image_encoder_name == "dinov2":
                features = self.image_encoder(predict_image_rect, output_hidden_states=True)
                features = features.last_hidden_state[:, 1:]
                features_hw = ratio_length2hw(predict_image_rect.shape[2:], length=features.size(1))
                features = rearrange(features, "b (h w) c -> b c h w", h=features_hw[0], w=features_hw[1])
            elif self.config.image_encoder_name == "dinov3":
                features = self.image_encoder(predict_image_rect)
                features = features["feature_2d"]
            elif self.config.image_encoder_name == "mars":
                features = features
            else:
                features = self.image_encoder(predict_image_rect)
            features_hw = features.shape[2:]
            outpainting_mask_rect = 1 - inpainting_mask_rect
            if self.config.use_gan:
                inpainting_and_out_mask_rect = torch.cat([inpainting_mask_rect, outpainting_mask_rect], dim=1)
                inpainting_and_out_mask_feature, _ = self.cal_masked_mean_feature(features, inpainting_and_out_mask_rect)
                feature_in_inpaint, feature_out_inpaint = inpainting_and_out_mask_feature.chunk(2, dim=2)
                feature_tokens = rearrange(features, "n c h w -> n (h w) c")
                feature_tokens = torch.cat([feature_in_inpaint.transpose(1, 2), feature_out_inpaint.transpose(1, 2), feature_tokens], dim=1)
                prob, gan_features = self.gan_transformers(feature_tokens)
                gan_loss = F.mse_loss(prob[..., 0], labels, reduction="none")
                gan_features = rearrange(gan_features[:, 2:].detach(), "n (h w) c -> n c h w", h=features_hw[0], w=features_hw[1])
                remove_loss['gan_loss'] = gan_loss
                return remove_loss, gan_features, cluster_entity
            else:
                if self.config.use_entity_similar:
                    use_predict_entity = self.config.use_predict_entity
                    if not use_predict_entity:
                        assert entity_mask_rect.size(0) == 1
                        entity_mask_rect, entity_indices = one_hot(entity_mask_rect[:, 0].long(), use_unique=True)
                        entity_mask_rect = entity_mask_rect[:, 1:]
                        entity_indices = entity_indices[1:]
                    else:
                        entity_mask_rect = F.interpolate(pred_masks_filter_sundries[None], size=inpainting_mask_rect.shape[2:], mode="bilinear") > 0
                    if entity_mask_rect.size(1) == 0:
                        remove_loss_entity = predict_image_rect.new_tensor(0)
                    else:
                        if not use_predict_entity:
                            with torch.no_grad():
                                entity_mask_rect_outpaint = entity_mask_rect * outpainting_mask_rect
                                entity_outpaint_feature, mask_sum = self.cal_masked_mean_feature(features, entity_mask_rect_outpaint)
                                valid = mask_sum > 0
                                entity_outpaint_feature = F.normalize(entity_outpaint_feature, dim=1)
                                features_norm = F.normalize(features, dim=1)
                                features_to_entity_outpaint_feature_similar = einsum(entity_outpaint_feature, features_norm, "n c k, n c h w -> n k h w")
                                to_entity_indices = features_to_entity_outpaint_feature_similar.argmax(1)
                                cluster_entity = entity_indices[to_entity_indices]
                                to_indices_one_hot, to_indices_one_hot_index = one_hot(to_entity_indices, use_unique=True)
                                entity_outpaint_feature_to_entity = entity_outpaint_feature[:, :, to_indices_one_hot_index]
                                to_indices_one_hot_inpaint = to_indices_one_hot * self.resize_mask(inpainting_mask_rect, features_hw)
                                to_indices_one_hot_inpaint_feature, _ = self.cal_masked_mean_feature(features, to_indices_one_hot_inpaint)
                                to_indices_one_hot_inpaint_feature = F.normalize(to_indices_one_hot_inpaint_feature, dim=1)
                                entity_similar = einsum(to_indices_one_hot_inpaint_feature, entity_outpaint_feature_to_entity, "n c k, n c k -> n k")
                                remove_loss_entity = 1 - entity_similar
                        else:
                            def get_entity_similar_loss(entity_mask_rect):
                                entity_mask_rect_inpaint = entity_mask_rect * inpainting_mask_rect
                                entity_mask_rect_outpaint = entity_mask_rect * outpainting_mask_rect
                                valid = (entity_mask_rect_inpaint.flatten(-2, -1) > 0).any(-1) & (entity_mask_rect_outpaint.flatten(-2, -1) > 0).any(-1)
                                if not valid.any():
                                    remove_loss_entity = predict_image_rect.new_tensor(0)
                                else:
                                    entity_mask_rect_inpaint = entity_mask_rect_inpaint[valid][None]
                                    entity_mask_rect_outpaint = entity_mask_rect_outpaint[valid][None]
                                    if self.config.use_mean_loss:
                                        entity_mask_rect_inoutpaint = torch.cat([entity_mask_rect_inpaint, entity_mask_rect_outpaint], dim=1)
                                        entity_inoutpaint_feature, mask_sum = self.cal_masked_mean_feature(features, entity_mask_rect_inoutpaint)
                                        assert (mask_sum > 0).all()
                                        # self.resize_mask(entity_mask_rect_inoutpaint, features_hw)
                                        entity_inpaint_feature, entity_outpaint_feature = torch.chunk(entity_inoutpaint_feature, 2, dim=-1)
                                        remove_loss_entity = 1 - F.cosine_similarity(entity_inpaint_feature.float(), entity_outpaint_feature.detach().float(), dim=1)
                                        mask_sum_inpaint = mask_sum.chunk(2, dim=-1)[0]
                                        remove_loss_entity = (remove_loss_entity * (mask_sum_inpaint / mask_sum_inpaint.sum().clamp(min=1e-4))).sum()
                                    else:
                                        entity_outpaint_feature, mask_sum = self.cal_masked_mean_feature(features, entity_mask_rect_outpaint)
                                        entity_outpaint_feature = entity_outpaint_feature.detach().float()
                                        entity_outpaint_feature = F.normalize(entity_outpaint_feature, p=2, dim=1)
                                        entity_dense_inpaint_feature = F.normalize(features.float(), p=2, dim=1)
                                        remove_loss_entity = 1 - einsum(entity_dense_inpaint_feature, entity_outpaint_feature, "n c h w, n c k -> n k h w" )
                                        # remove_loss_entity = 1 - F.cosine_similarity(entity_dense_inpaint_feature, entity_outpaint_feature, dim=1)
                                        entity_mask_rect_inpaint_resized = self.resize_mask(entity_mask_rect_inpaint, features_hw)
                                        entity_mask_rect_inpaint_resized_sum1 = entity_mask_rect_inpaint_resized.sum(1)
                                        remove_loss_entity = (remove_loss_entity * entity_mask_rect_inpaint_resized).sum(1) / entity_mask_rect_inpaint_resized_sum1.clamp(min=1e-4)
                                        entity_mask_rect_inpaint_resized_sum2 = entity_mask_rect_inpaint_resized_sum1 / entity_mask_rect_inpaint_resized_sum1.clamp(min=1e-4)
                                        remove_loss_entity = remove_loss_entity.sum() / entity_mask_rect_inpaint_resized_sum2.sum().clamp(min=1e-4)
                                return remove_loss_entity
                            if self.config.use_teacher_entity \
                                    or (self.config.use_teacher_entity_on_pair_data and data_tag == "pair_label"):
                                pred_mars_masks_teacher = F.interpolate(mars_output_teacher.pred_mars_masks[i_image][None, None], size=entity_mask_rect.shape[2:], mode="bilinear") > 0
                                pred_masks_filter_sundries_teacher = mars_output_teacher.pred_masks_filter_sundries
                                entity_mask_rect_teacher = F.interpolate(pred_masks_filter_sundries_teacher[i_image][None], size=entity_mask_rect.shape[2:], mode="bilinear") > 0
                                if self.config.only_teacher_entity:
                                    entity_mask_rect = entity_mask_rect_teacher
                                else:
                                    entity_mask_rect = entity_mask_rect & pred_mars_masks_teacher
                                    entity_mask_rect = torch.cat([entity_mask_rect, entity_mask_rect_teacher], dim=1)

                            remove_loss_entity = get_entity_similar_loss(entity_mask_rect)
                            cluster_entity_, cluster_entity = entity_mask_rect.max(1); cluster_entity += 1; cluster_entity[cluster_entity_ == 0] = 0

                    remove_loss['remove_loss_entity'] = remove_loss_entity

        return remove_loss, features, cluster_entity

    def forward(self, predict_images, inpainting_masks, entity_masks, *,
                pred_masks_filter_sundries=None, labels=None, mars_loss_output, mars_output_teacher, data_tag):
        """
        predict_image: N C H W
        inpainting_mask: N 1 H W
        """
        inpainting_masks = inpainting_masks[:, 0]
        if entity_masks is not None:
            entity_masks = entity_masks[:, 0]
        bboxes = masks2bboxes(inpainting_masks)
        image_shape = predict_images.shape[2:]
        if self.config.use_crop:
            predict_image_rect = []
            inpainting_mask_rect = []
            entity_mask_rect = []
            pred_masks_filter_sundries_rect = []
            mask_features_rect = []
            if self.config.image_encoder_name == "mars":
                mask_features = F.interpolate(mars_loss_output.mask_features, size=image_shape, mode="nearest-exact")
            for i_image, (predict_image, inpainting_mask, bbox) in enumerate(zip(predict_images, inpainting_masks, bboxes)):
                if self.config.use_entity_similar:
                    entity_mask = entity_masks[i_image]
                    pred_masks_filter_sundries_ = F.interpolate(pred_masks_filter_sundries[i_image][None], size=predict_images.shape[2:], mode='bilinear', align_corners=False)[0]
                if self.config.image_encoder_name == "mars":
                    mask_feature = mask_features[i_image]
                bbox = expand_box(bbox, ratio=self.expand_ratio, height=image_shape[0], width=image_shape[1])
                bbox = bbox.round().long()
                if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                    predict_image = torch.zeros_like(predict_image)
                    inpainting_mask = torch.zeros_like(inpainting_mask)
                    if self.config.use_entity_similar:
                        entity_mask = torch.zeros_like(entity_mask)
                        pred_masks_filter_sundries_ = pred_masks_filter_sundries_.new_zeros([0, *pred_masks_filter_sundries_.shape[1:]])
                    if self.config.image_encoder_name == "mars":
                        mask_feature = torch.zeros_like(mask_feature)
                        # mask_feature = F.interpolate(mask_feature[None], size=predict_images.shape[2:], mode='bilinear', align_corners=False)[0]
                else:
                    predict_image = predict_image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    inpainting_mask = inpainting_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if self.config.use_entity_similar:
                        entity_mask = entity_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        pred_masks_filter_sundries_ = pred_masks_filter_sundries_[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if self.config.image_encoder_name == "mars":
                        mask_feature = mask_feature[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if self.config.image_encoder_name == "sam":
                    target_size = self.predictor.transform.get_preprocess_shape(image_shape[0], image_shape[1], self.predictor.transform.target_length)
                elif self.config.image_encoder_name in {"dinov2", "efficient_sam"}:
                    target_size = self.config.image_size
                elif self.config.image_encoder_name == "mars":
                    target_size = image_shape
                predict_image = F.interpolate(predict_image[None], size=target_size, mode="bilinear", align_corners=False, antialias=True)[0]
                inpainting_mask = F.interpolate(inpainting_mask[None, None], size=target_size, mode="nearest-exact")[0, 0]
                if self.config.image_encoder_name == "sam":
                    predict_image = self.predictor.model.preprocess_scale1(predict_image)
                    inpainting_mask = self.predictor.model.preprocess_pad(inpainting_mask)
                else:
                    predict_image = self.pre_norm(predict_image[None])[0]
                if self.config.use_entity_similar:
                    entity_mask = F.interpolate(entity_mask[None, None], size=target_size, mode="nearest-exact")[0, 0]
                    entity_mask_rect.append(entity_mask)
                    pred_masks_filter_sundries_rect.append(pred_masks_filter_sundries_)
                predict_image_rect.append(predict_image)
                inpainting_mask_rect.append(inpainting_mask)
                if self.config.image_encoder_name == "mars":
                    mask_feature = F.interpolate(mask_feature[None], size=target_size, mode="nearest-exact")[0]
                    mask_features_rect.append(mask_feature)

            predict_image_rect = torch.stack(predict_image_rect, dim=0)
            inpainting_mask_rect = torch.stack(inpainting_mask_rect, dim=0)[:, None]
            if self.config.use_entity_similar:
                entity_mask_rect = torch.stack(entity_mask_rect, dim=0)[:, None]
            else:
                entity_mask_rect = None
            if self.config.image_encoder_name == "mars":
                mask_features = torch.stack(mask_features_rect, dim=0)
            else:
                mask_features = None
        else:
            predict_image_rect = predict_images
            inpainting_mask_rect = inpainting_masks
            entity_mask_rect = entity_masks
            pred_masks_filter_sundries_rect = pred_masks_filter_sundries
            if self.config.image_encoder_name in {"dinov2", "efficient_sam", "dinov3"}:
                target_size = self.config.image_size
            if self.config.image_encoder_name == "mars":
                mask_features = mars_loss_output.mask_features
                inpainting_mask_rect = inpainting_mask_rect[:, None]
            else:
                predict_image_rect = F.interpolate(predict_image_rect, size=target_size, mode="bilinear", align_corners=False, antialias=True)
                inpainting_mask_rect = F.interpolate(inpainting_mask_rect[:, None], size=target_size, mode="nearest-exact")
            predict_image_rect = self.pre_norm(predict_image_rect)

        if not self.config.backward_respectively:
            remove_loss, features = self.get_loss(predict_image_rect, inpainting_mask_rect)
            predict_image_rect_grad = None
        else:
            # for save memory
            # predict_image_rect = predict_image_rect.detach(); predict_image_rect.requires_grad_(True)
            features_lt = []
            remove_loss_lt = dict(remove_loss_global=[], remove_loss_entity=[], gan_loss=[])
            predict_image_rect_grad = []
            cluster_entities = []
            batch_size = len(predict_image_rect)
            for i_image, (predict_image_rect_, inpainting_mask_) in enumerate(zip(predict_image_rect, inpainting_mask_rect)):
                predict_image_rect_: torch.Tensor = predict_image_rect_.clone().detach()[None]
                inpainting_mask_ = inpainting_mask_.clone().detach()[None]
                predict_image_rect_.requires_grad_(True)
                if entity_mask_rect is not None:
                    entity_mask_rect_ = entity_mask_rect[i_image:i_image + 1]
                else:
                    entity_mask_rect_ = None
                if labels is not None:
                    labels_ = labels[i_image:i_image + 1]
                else:
                    labels_ = None
                if self.config.use_entity_similar:
                    pred_masks_filter_sundries_ = pred_masks_filter_sundries_rect[i_image] if pred_masks_filter_sundries_rect is not None else None
                else:
                    pred_masks_filter_sundries_ = None
                if self.config.image_encoder_name == "mars":
                    mask_features_ = mask_features[i_image:i_image + 1]
                else:
                    mask_features_ = None
                remove_loss, feature, cluster_entity = self.get_loss(predict_image_rect_, inpainting_mask_, entity_mask_rect_, pred_masks_filter_sundries_, labels=labels_, features=mask_features_, i_image=i_image, mars_output_teacher=mars_output_teacher, data_tag=data_tag)
                features_lt.append(feature)
                if self.config.image_encoder_name == "mars":
                    for key, value in remove_loss.items():
                        remove_loss_lt[key].append(value.mean())
                else:
                    loss_ = 0
                    for value in remove_loss.values():
                        loss_ += value.mean()
                    (loss_ / batch_size).backward()
                    for key, value in remove_loss.items():
                        remove_loss_lt[key].append(value.detach().mean())
                    predict_image_rect_grad.append(predict_image_rect_.grad)
                if cluster_entity is not None:
                    cluster_entities.append(cluster_entity)
            if self.config.image_encoder_name == "mars":
                predict_image_rect_grad = None
            else:
                predict_image_rect_grad = torch.cat(predict_image_rect_grad, dim=0)
                predict_image_rect_grad = fix_grad_forward(predict_image_rect, predict_image_rect_grad)
                # predict_image_rect.backward(gradient=predict_image_rect_grad)
                # predict_image_rect_grad = F.mse_loss(predict_image_rect, (predict_image_rect.detach() - predict_image_rect_grad)) * 0.5

            remove_loss_lt_ = {}
            for key, value in remove_loss_lt.items():
                if len(value) > 0:
                    remove_loss_lt_[key] = torch.stack(value, dim=0)
            remove_loss_lt = remove_loss_lt_
            features = torch.cat(features_lt, dim=0)
            if len(cluster_entities):
                cluster_entities = torch.cat(cluster_entities, dim=0)
            else:
                cluster_entities = None

        return RemoveLossOutput(loss=remove_loss_lt, image_features=features, predict_image_rect=predict_image_rect, predict_image_rect_grad=predict_image_rect_grad, cluster_entities=cluster_entities, inpainting_mask_rect=inpainting_mask_rect, entity_mask_rect=entity_mask_rect)

if __name__ == '__main__':
    import cv2
    from unhcv.common.image import masks2bboxes, ratio_length2hw, visual_feature
    # remove_loss = RemoveLoss(dict(image_encoder_name="dinov2")).cuda()
    remove_loss = RemoveLoss(dict(image_encoder_name="efficient_sam", image_size=1024, backward_respectively=True, use_gan=True)).cuda()
    predict_images = torch.randn(2, 3, 512, 256).cuda()
    predict_images.requires_grad_(True)
    inpainting_masks = torch.zeros(2, 1, 512, 256).cuda()
    inpainting_masks[:, 0, 20:30, 40:50] = 1
    entity_masks = torch.zeros(2, 1, 512, 256).cuda()
    entity_masks[:, 0, 15:25, 42] = 1
    entity_masks[:, 0, 15:25, 41] = 2
    labels = torch.ones([2]).cuda()
    output = remove_loss.forward(predict_images, inpainting_masks, entity_masks, labels=labels)

    # predict_images = obj_load('/home/yixing/code/EfficientSAM/figs/examples/for_seg.png').convert('RGB')
    # predict_images = to_tensor(predict_images).cuda()[None]
    # predict_images.requires_grad_(True)
    # inpainting_masks = torch.ones_like(predict_images)
    # predict_images = predict_images.repeat(2, 1, 1, 1)
    # inpainting_masks = inpainting_masks.repeat(2, 1, 1, 1)
    # inpainting_masks[1] = 0

    # tmp = torch.load("/XYFS01/sysu_qingzhang_1/train_outputs/tmp1.bin")
    # predict_images = tmp["predict_images"] #.float()
    # inpainting_masks = tmp["inpainting_masks"][:, None]# .float()
    # features_lt_ = tmp["features_lt"]
    # remove_loss = remove_loss.to(torch.bfloat16)

    # def method1():
    #     output0 = output = remove_loss.forward(predict_images, inpainting_masks)
    #     loss = output['predict_image_rect_grad'].sum(); loss.backward()
    #     predict_images.grad = None
    #
    # print(timeit.timeit(method1, number=10))
    output = remove_loss.forward(predict_images, inpainting_masks, entity_masks)
    show_feature = visual_feature(output.image_features)
    cv2.imwrite('/home/yixing/train_outputs/test7.png', show_feature)
    # output['predict_image_rect_grad'].backward(grad_tensors=)
    grad0 = predict_images.grad
    def method2():
        remove_loss.config.backward_respectively = False
        output1 = output = remove_loss.forward(predict_images, inpainting_masks)
        output.loss.mean().backward()
        predict_images.grad = None

    print(timeit.timeit(method2, number=10))
    grad1 = predict_images.grad
    monitor_memory()
    pass
