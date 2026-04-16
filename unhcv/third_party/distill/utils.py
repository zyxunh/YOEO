import numpy as np
import torch
from typing import Optional, Union, List

from diffusers import LCMScheduler

from unhcv.common.utils import get_logger


logger = get_logger(__name__)


__all__ = ["CustomLCMScheduler"]


class CustomLCMScheduler(LCMScheduler):

    timesteps_dict: dict = {1: (999,), 2: (999, 749), 3: (999, 749, 499), 4: (999, 749, 499, 249)}

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012,
                 beta_schedule: str = "scaled_linear", trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
                 original_inference_steps: int = 50, clip_sample: bool = False, clip_sample_range: float = 1.0,
                 set_alpha_to_one: bool = True, steps_offset: int = 0, prediction_type: str = "epsilon",
                 thresholding: bool = False, dynamic_thresholding_ratio: float = 0.995, sample_max_value: float = 1.0,
                 timestep_spacing: str = "leading", timestep_scaling: float = 10.0,
                 rescale_betas_zero_snr: bool = False):
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule, trained_betas,
                         original_inference_steps, clip_sample, clip_sample_range, set_alpha_to_one, steps_offset,
                         prediction_type, thresholding, dynamic_thresholding_ratio, sample_max_value, timestep_spacing,
                         timestep_scaling, rescale_betas_zero_snr)


    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        original_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        strength: int = 1.0,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`, *optional*):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            original_inference_steps (`int`, *optional*):
                The original number of inference steps, which will be used to generate a linearly-spaced timestep
                schedule (which is different from the standard `diffusers` implementation). We will then take
                `num_inference_steps` timesteps from this schedule, evenly spaced in terms of indices, and use that as
                our final timestep schedule. If not set, this will default to the `original_inference_steps` attribute.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps on the training/distillation timestep
                schedule is used. If `timesteps` is passed, `num_inference_steps` must be `None`.
        """
        # 0. Check inputs
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `custom_timesteps`.")

        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        # 1. Calculate the LCM original training/distillation timestep schedule.
        original_steps = (
            original_inference_steps if original_inference_steps is not None else self.config.original_inference_steps
        )

        if original_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`original_steps`: {original_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        # LCM Timesteps Setting
        # The skipping step parameter k from the paper.
        k = self.config.num_train_timesteps // original_steps
        # LCM Training/Distillation Steps Schedule
        # Currently, only a linearly-spaced schedule is supported (same as in the LCM distillation scripts).
        lcm_origin_timesteps = np.asarray(list(range(1, int(original_steps * strength) + 1))) * k - 1

        timesteps = self.timesteps_dict[num_inference_steps]
        timesteps = np.array(timesteps, dtype=np.int64)
        self.num_inference_steps = len(timesteps)
        self.custom_timesteps = True

        # 2. Calculate the LCM inference timestep schedule.
        # if timesteps is not None:
        #     # 2.1 Handle custom timestep schedules.
        #     train_timesteps = set(lcm_origin_timesteps)
        #     non_train_timesteps = []
        #     for i in range(1, len(timesteps)):
        #         if timesteps[i] >= timesteps[i - 1]:
        #             raise ValueError("`custom_timesteps` must be in descending order.")
        #
        #         if timesteps[i] not in train_timesteps:
        #             non_train_timesteps.append(timesteps[i])
        #
        #     if timesteps[0] >= self.config.num_train_timesteps:
        #         raise ValueError(
        #             f"`timesteps` must start before `self.config.train_timesteps`:"
        #             f" {self.config.num_train_timesteps}."
        #         )
        #
        #     # Raise warning if timestep schedule does not start with self.config.num_train_timesteps - 1
        #     if strength == 1.0 and timesteps[0] != self.config.num_train_timesteps - 1:
        #         logger.warning(
        #             f"The first timestep on the custom timestep schedule is {timesteps[0]}, not"
        #             f" `self.config.num_train_timesteps - 1`: {self.config.num_train_timesteps - 1}. You may get"
        #             f" unexpected results when using this timestep schedule."
        #         )
        #
        #     # Raise warning if custom timestep schedule contains timesteps not on original timestep schedule
        #     if non_train_timesteps:
        #         logger.warning(
        #             f"The custom timestep schedule contains the following timesteps which are not on the original"
        #             f" training/distillation timestep schedule: {non_train_timesteps}. You may get unexpected results"
        #             f" when using this timestep schedule."
        #         )
        #
        #     # Raise warning if custom timestep schedule is longer than original_steps
        #     if len(timesteps) > original_steps:
        #         logger.warning(
        #             f"The number of timesteps in the custom timestep schedule is {len(timesteps)}, which exceeds the"
        #             f" the length of the timestep schedule used for training: {original_steps}. You may get some"
        #             f" unexpected results when using this timestep schedule."
        #         )
        #
        #     timesteps = np.array(timesteps, dtype=np.int64)
        #     self.num_inference_steps = len(timesteps)
        #     self.custom_timesteps = True
        #
        #     # Apply strength (e.g. for img2img pipelines) (see StableDiffusionImg2ImgPipeline.get_timesteps)
        #     init_timestep = min(int(self.num_inference_steps * strength), self.num_inference_steps)
        #     t_start = max(self.num_inference_steps - init_timestep, 0)
        #     timesteps = timesteps[t_start * self.order :]
        #     # TODO: also reset self.num_inference_steps?
        # else:
        #     # 2.2 Create the "standard" LCM inference timestep schedule.
        #     if num_inference_steps > self.config.num_train_timesteps:
        #         raise ValueError(
        #             f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
        #             f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
        #             f" maximal {self.config.num_train_timesteps} timesteps."
        #         )
        #
        #     skipping_step = len(lcm_origin_timesteps) // num_inference_steps
        #
        #     if skipping_step < 1:
        #         raise ValueError(
        #             f"The combination of `original_steps x strength`: {original_steps} x {strength} is smaller than `num_inference_steps`: {num_inference_steps}. Make sure to either reduce `num_inference_steps` to a value smaller than {int(original_steps * strength)} or increase `strength` to a value higher than {float(num_inference_steps / original_steps)}."
        #         )
        #
        #     self.num_inference_steps = num_inference_steps
        #
        #     if num_inference_steps > original_steps:
        #         raise ValueError(
        #             f"`num_inference_steps`: {num_inference_steps} cannot be larger than `original_inference_steps`:"
        #             f" {original_steps} because the final timestep schedule will be a subset of the"
        #             f" `original_inference_steps`-sized initial timestep schedule."
        #         )
        #
        #     # LCM Inference Steps Schedule
        #     lcm_origin_timesteps = lcm_origin_timesteps[::-1].copy()
        #     # Select (approximately) evenly spaced indices from lcm_origin_timesteps.
        #     inference_indices = np.linspace(0, len(lcm_origin_timesteps), num=num_inference_steps, endpoint=False)
        #     inference_indices = np.floor(inference_indices).astype(np.int64)
        #     timesteps = lcm_origin_timesteps[inference_indices]

        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.long)

        self._step_index = None
        self._begin_index = None
