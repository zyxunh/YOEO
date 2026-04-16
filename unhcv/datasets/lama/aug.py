from albumentations import DualIAATransform, to_tuple
# from albumentations.imgaug.transforms import DualIAATransform
import imgaug.augmenters as iaa
import albumentations as A
from unhcv.common.image.visual import visual_tensor

class IAAAffine2(DualIAATransform):
    """Place a regular grid of points on the input and randomly move the neighbourhood of these point around
    via affine transformations.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(
        self,
        scale=(0.7, 1.3),
        translate_percent=None,
        translate_px=None,
        rotate=0.0,
        shear=(-0.1, 0.1),
        order=1,
        cval=0,
        mode="reflect",
        always_apply=False,
        p=0.5,
    ):
        super(IAAAffine2, self).__init__(always_apply, p)
        self.scale = dict(x=scale, y=scale)
        self.translate_percent = to_tuple(translate_percent, 0)
        self.translate_px = to_tuple(translate_px, 0)
        self.rotate = to_tuple(rotate)
        self.shear = dict(x=shear, y=shear)
        self.order = order
        self.cval = cval
        self.mode = mode

    @property
    def processor(self):
        return iaa.Affine(
            self.scale,
            self.translate_percent,
            self.translate_px,
            self.rotate,
            self.shear,
            self.order,
            self.cval,
            self.mode,
        )

    def get_transform_init_args_names(self):
        return ("scale", "translate_percent", "translate_px", "rotate", "shear", "order", "cval", "mode")


class IAAPerspective2(DualIAATransform):
    """Perform a random four point perspective transform of the input.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        scale ((float, float): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners. Default: (0.05, 0.1).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(self, scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5,
                 order=1, cval=0, mode="replicate"):
        super(IAAPerspective2, self).__init__(always_apply, p)
        self.scale = to_tuple(scale, 1.0)
        self.keep_size = keep_size
        self.cval = cval
        self.mode = mode

    @property
    def processor(self):
        return iaa.PerspectiveTransform(self.scale, keep_size=self.keep_size, mode=self.mode, cval=self.cval)

    def get_transform_init_args_names(self):
        return ("scale", "keep_size")


if __name__ == "__main__":
    def get_transforms(transform_variant, out_size):
        if transform_variant == 'default':
            transform = A.Compose([
                A.RandomScale(scale_limit=0.2),  # +/- 20%
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
                A.ToFloat()
            ])
        elif transform_variant == 'distortions':
            transform = A.Compose([
                IAAPerspective2(scale=(0.0, 0.06)),
                IAAAffine2(scale=(0.7, 1.3),
                        rotate=(-40, 40),
                        shear=(-0.1, 0.1)),
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.OpticalDistortion(),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
                A.ToFloat()
            ])
        elif transform_variant == 'distortions_scale05_1':
            transform = A.Compose([
                IAAPerspective2(scale=(0.0, 0.06)),
                IAAAffine2(scale=(0.5, 1.0),
                        rotate=(-40, 40),
                        shear=(-0.1, 0.1),
                        p=1),
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.OpticalDistortion(),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
                A.ToFloat()
            ])
        elif transform_variant == 'distortions_scale03_12':
            transform = A.Compose([
                IAAPerspective2(scale=(0.0, 0.06)),
                IAAAffine2(scale=(0.3, 1.2),
                        rotate=(-40, 40),
                        shear=(-0.1, 0.1),
                        p=1),
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.OpticalDistortion(),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
                A.ToFloat()
            ])
        elif transform_variant == 'distortions_scale03_07':
            transform = A.Compose([
                IAAPerspective2(scale=(0.0, 0.06)),
                IAAAffine2(scale=(0.3, 0.7),  # scale 512 to 256 in average
                        rotate=(-40, 40),
                        shear=(-0.1, 0.1),
                        p=1),
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.OpticalDistortion(),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
                A.ToFloat()
            ])
        elif transform_variant == 'distortions_light':
            transform = A.Compose([
                IAAPerspective2(scale=(0.0, 0.02)),
                IAAAffine2(scale=(0.8, 1.8),
                        rotate=(-20, 20),
                        shear=(-0.03, 0.03)),
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
                A.ToFloat()
            ])
        elif transform_variant == 'non_space_transform':
            transform = A.Compose([
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
                A.ToFloat()
            ])
        elif transform_variant == 'no_augs':
            transform = A.Compose([
                A.ToFloat()
            ])
        elif transform_variant == 'unh':
            transform = A.Compose([
                IAAPerspective2(scale=(0.0, 0.06)),
                IAAAffine2(scale=(0.3, 1.2),
                        rotate=(-40, 40),
                        shear=(-0.1, 0.1),
                        p=1),
                # A.PadIfNeeded(min_height=out_size, min_width=out_size),
                # A.OpticalDistortion(),
                # A.RandomCrop(height=out_size, width=out_size),
                # A.HorizontalFlip(),
            ])
        else:
            raise ValueError(f'Unexpected transform_variant {transform_variant}')
        return transform
    import cv2
    file_name = "/home/tiger/tmp/debug/image/679ad92a529b6ba5_144.jpg"
    im = cv2.imread(file_name)
    transformer = get_transforms("unh", 512)
    out = transformer(image=im)['image']
    cv2.imwrite('/home/tiger/code/lama/im.jpg', im)
    cv2.imwrite('/home/tiger/code/lama/out.jpg', out)
    breakpoint()
    pass