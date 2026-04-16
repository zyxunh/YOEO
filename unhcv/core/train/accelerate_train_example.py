import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union, Dict, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset

from unhcv.common import write_im, CfgNode
from unhcv.common.image import concat_differ_size
from unhcv.common.types import default_factory
from unhcv.core.train import TrainConfig, AccelerateTrainer


@dataclass(repr=False)
class DefaultDatasetConfig(CfgNode):
    num_samples: int = 100
    tensor_shape: tuple = (3, 64, 64)


class FixedTensorDataset(Dataset):
    def __init__(self, config: DefaultDatasetConfig=DefaultDatasetConfig()):
        """
        Toy Dataset 生成固定形状的随机 Tensor
        参数:
            num_samples: 数据集样本数 (默认: 100)
            tensor_shape: 生成 Tensor 的形状 (默认: 3x64x64 模拟图像)
            num_classes: 分类类别数 (默认: 10)
        """
        self.num_samples = config.num_samples
        self.tensor_shape = config.tensor_shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return dict(x=torch.zeros(self.tensor_shape) + idx)


@dataclass
class ExampleConfig(TrainConfig):
    default_dataset_kwargs: Optional[Union[Dict, Sequence[Dict]]] = default_factory(DefaultDatasetConfig())
    dataset_class: Optional[type] = FixedTensorDataset
    default_test_dataset_kwargs: Optional[Union[Dict, Sequence[Dict]]] = default_factory(DefaultDatasetConfig())
    test_dataset_class: Optional[type] = FixedTensorDataset


class ExampleTrainer(AccelerateTrainer):

    def __init__(self, config: ExampleConfig=ExampleConfig()):
        super().__init__(config)

    def init_model(self):
        super().init_model()
        model_config = self.parse_model_config()
        model = nn.Conv2d(3, 100, 3, 1, 0)
        self.model = (model,)

    def get_loss(self, batch: Any):
        x = batch['x'].to(self.weight_dtype).cuda()
        out = self.model[0](x)
        loss = out.mean() * 0
        return dict(loss_toy=loss)

    def visual_train_dataset(self, data, batch_index):
        show_dir = os.path.join(self.args.show_dir, 'train_data')
        write_im(os.path.join(show_dir, "{:05}.jpg".format(batch_index)), shows)
        pass

    def visual_training_result(self):
        show_root = os.path.join(self.args.show_dir, 'training/{:06}'.format(self.global_step))
        save_for_training_show_tensors = self.save_for_training_show_tensors
        model = self.accelerator.unwrap_model(self.model[0])
        model.eval()
        with torch.no_grad():
            output = model(model_input)
        shows = concat_differ_size(shows, axis=0)
        write_im(os.path.join(show_root, "show.jpg"), shows)
        model.train()

    def parse_train_dataset_config(self):
        dataset_config = self.parse_dataset_config()
        dataset_config["phase"] = "train"
        return dataset_config

    def parse_test_dataset_config(self):
        dataset_config = self.parse_dataset_config()
        dataset_config["phase"] = "test"
        return dataset_config

    def inference_on_test(self, global_step):
        test_dataloaders = self.init_test_dataset(test_for_which="test")
        if not isinstance(test_dataloaders, tuple):
            test_dataloaders = (test_dataloaders,)
        tests = []
        for test_dataloader in test_dataloaders:
            for data in test_dataloader:
                with torch.no_grad():
                    tests.append(data['x'].unique().cuda())
        tests = torch.stack(tests)
        x = self.accelerator.gather_for_metrics(tests, use_gather_object=True)
        # x1 = all_gather_object([self.accelerator.local_process_index])
        # print(x1)
        x = [var.to(self.accelerator.device) for var in x]
        tests = torch.cat(x)
        print(tests)
        print(tests.unique().shape)
        print('done: {}'.format(self.accelerator.process_index))

    def inference_on_train(self, global_step):
        pass

if __name__ == "__main__":
    trainer = ExampleTrainer()
    trainer.main()
    pass
