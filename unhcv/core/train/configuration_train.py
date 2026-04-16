from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Sequence

import torch

from unhcv.common import CfgNode
from unhcv.common.types import default_factory


@dataclass(repr=False)
class SchedulerConfig(CfgNode):
    name: str = None
    final_ratio: float = 0.01

@dataclass(repr=False)
class OptimConfig(CfgNode):
    name: str = "AdamW"
    learning_rate: float = None

@dataclass(repr=False)
class TrainConfig(CfgNode):
    """
    Configuration class for training.

    Attributes:
        default_dataset_kwargs (Optional[Union[Dict, Sequence[Dict]]]): Default keyword arguments for the dataset.
        default_demo_dataset_kwargs (Optional[Dict]): Default keyword arguments for the demo dataset.
        dataset_class (Optional[type]): Class type for the dataset.
        max_visual_num_in_training (int): Maximum number of visual elements in training. Default is 5.
        demo_dataset_class (Optional[type]): Class type for the demo dataset.
        prepare_train_dataloader (bool): Flag to prepare the training DataLoader. Default is True.
        demo_batch_size (int): Batch size for the demo dataset. Default is 1.
    """
    dataset_class: Union[type, str] = None
    default_dataset_kwargs: Optional[Union[Dict, Sequence[Dict]]] = default_factory(CfgNode())
    dataset_collate_fn: type = torch.utils.data.default_collate
    demo_dataset_class: Optional[type] = None
    default_demo_dataset_kwargs: Optional[Dict] = default_factory({})
    test_dataset_class: Optional[type] = None
    default_test_dataset_kwargs: Optional[Union[Dict, Sequence[Dict]]] = default_factory(CfgNode())
    max_visual_num_in_training: int = 5
    demo_batch_size: int = 1
    scheduler_config: SchedulerConfig = default_factory(SchedulerConfig)
    optim_config: OptimConfig = default_factory(OptimConfig)

    effective_batch_size: int = None
    train_batch_size: int = None
    train_steps: int = None



if __name__ == "__main__":
    pass