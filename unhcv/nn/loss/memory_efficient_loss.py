import torch
import torch.nn.functional as F


class PointSampleLoss(torch.nn.Module):
    def __init__(self, sample_mode="nearest", num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75):
        super().__init__()
        self.sample_mode = sample_mode
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def point_sample(self, input, point_coords, **kwargs):
        """
        A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
        Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
        [0, 1] x [0, 1] square.

        Args:
            input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
            point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
            [0, 1] x [0, 1] normalized point coordinates.

        Returns:
            output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
                features for points in `point_coords`. The features are obtained via bilinear
                interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
        """
        add_dim = False
        if point_coords.dim() == 3:
            add_dim = True
            point_coords = point_coords.unsqueeze(2)
        output = F.grid_sample(input, 2.0 * point_coords - 1.0, mode=self.sample_mode, **kwargs)
        if add_dim:
            output = output.squeeze(3)
        return output

    @staticmethod
    def calculate_uncertainty(logits):
        """
        We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
            foreground class in `classes`.
        Args:
            logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
                class-agnostic, where R is the total number of predicted masks in all images and C is
                the number of foreground classes. The values are logits.
        Returns:
            scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
                the most uncertain locations having the highest uncertainty score.
        """
        assert logits.shape[1] == 1
        gt_class_logits = logits.clone()
        return -(torch.abs(gt_class_logits))


    def get_uncertain_point_coords_with_randomness(
        self, coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
    ):
        """
        Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
            are calculated for each point using 'uncertainty_func' function that takes point's logit
            prediction as input.
        See PointRend paper for details.

        Args:
            coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
                class-specific or class-agnostic prediction.
            uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
                contains logit predictions for P points and returns their uncertainties as a Tensor of
                shape (N, 1, P).
            num_points (int): The number of points P to sample.
            oversample_ratio (int): Oversampling parameter.
            importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

        Returns:
            point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
                sampled points.
        """
        assert oversample_ratio >= 1
        assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
        num_boxes = coarse_logits.shape[0]
        num_sampled = int(num_points * oversample_ratio)
        point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
        point_logits = self.point_sample(coarse_logits, point_coords, align_corners=False)
        # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
        # Calculating uncertainties of the coarse predictions first and sampling them for points leads
        # to incorrect results.
        # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
        # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
        # However, if we calculate uncertainties for the coarse predictions first,
        # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
        point_uncertainties = uncertainty_func(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
        idx += shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
            num_boxes, num_uncertain_points, 2
        )
        if num_random_points > 0:
            point_coords = torch.cat(
                [
                    point_coords,
                    torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
                ],
                dim=1,
            )
        return point_coords

    def forward(self, src_masks, target_masks, valid_masks=None, loss_func=None, num_points=None, oversample_ratio=None, importance_sample_ratio=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # assert "pred_masks" in outputs
        #
        # src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)
        # src_masks = outputs["pred_masks"]
        # src_masks = src_masks[src_idx]
        # masks = [t["masks"] for t in targets]
        # # TODO use valid to mask invalid areas due to padding in loss
        # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        # target_masks = target_masks.to(src_masks)
        # target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        if num_points is None:
            num_points = self.num_points
        if oversample_ratio is None:
            oversample_ratio = self.oversample_ratio
        if importance_sample_ratio is None:
            importance_sample_ratio = self.importance_sample_ratio

        # src_masks = src_masks[:, None]
        # target_masks = target_masks[:, None]

        with torch.no_grad():
            if valid_masks is not None:
                src_masks.masked_fill_(valid_masks == 0, -1000)

            # sample point_coords
            point_coords = self.get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: self.calculate_uncertainty(logits),
                num_points,
                oversample_ratio,
                importance_sample_ratio,
            )
            # get gt labels
            point_labels = self.point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

            if valid_masks is not None:
                assert self.sample_mode == "nearest"
                valid_masks = self.point_sample(
                    valid_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)

        point_logits = self.point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        loss = F.binary_cross_entropy_with_logits(point_logits, point_labels, reduction='none')
        loss = (loss * valid_masks).sum() / valid_masks.sum().clamp(min=1)
        return loss


if __name__ == "__main__":
    point_sample_loss = PointSampleLoss()
    x = torch.randn(3, 1, 100, 100)
    y = torch.randn(3, 1, 100, 100)
    valid = (torch.randn(3, 1, 100, 100) > 0).to(x)
    loss1 = point_sample_loss(x, y, valid)
