import os
import random
import argparse
import json
import itertools
import time

import torch
import torch.distributed
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from typing import Any, Optional, Tuple, Union

from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPOutput, clip_loss
import torch
import torch.nn as nn
from unhcv.math import torch_corroef
from diffusers.models.attention import BasicTransformerBlock, FeedForward

from unhcv.projects.diffusion.data.dataset_inpaint_bucket import get_inpaint_bucket_dataset
from unhcv.projects.diffusion.data.dataset_clip import ClipDataset
from unhcv.common.utils.progressbar import ProgressBarTqdm
from unhcv.models.nn.utils import PreNorm
from unhcv.common.utils import remove_dir, obj_dump, obj_load, write_txt, human_format_num
from unhcv.core.train.accelerate_train import AccelerateTrainer
from unhcv.datasets.utils import FakeDataset
from torch.utils.data import DataLoader


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            *(nn.Linear(10000, 10000) for _ in range(10))
        )

    def forward(self, x):
        return self.linear(x)


class FSDPTrain(AccelerateTrainer):
    def init_model(self):
        self.model = Layer()

    def init_for_train(self):
        args = self.args
        accelerator = self.accelerator
        model = self.model
        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        dataset = FakeDataset(torch.ones(10000))
        dataloader = DataLoader(dataset, num_workers=1, batch_size=2)
        self.logger.info("***** Running training *****")
        # self.logger.info(
        #     f"world_size is {world_size}, batch_size is {world_size * args.train_batch_size}, num_epoch is {world_size * args.train_batch_size * args.train_steps / train_dataset.data_length}")

        # Prepare everything with our `accelerator`.
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        print(model)
        # print(self.model.state_dict().keys())
        train_dataiter = iter(dataloader)
        self.train_dataiter = train_dataiter
        # self.train_dataiter = accelerator.prepare(train_dataiter)
        self.model = model
        self.optimizer = optimizer

    def init_for_eval(self):
        raise NotImplementedError

    def get_loss(self, batch: Any):
        output = self.model(batch)
        loss = output.sum()
        return dict(loss=loss)

    def inference_on_train(self, global_step):
        pass

    def inference_on_test(self, global_step):
        pass

    def inference_on_demo(self, global_step):
        pass

    def inference(self, global_step):
        self.inference_on_demo(global_step=global_step)
        self.inference_on_test(global_step=global_step)
        self.inference_on_train(global_step=global_step)

    def init_accelerator(self):
        args = self.args
        logging_dir = os.path.join(args.output_dir, "log")

        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

        self.accelerator = accelerator = Accelerator(
            # mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )
        accelerator.init_trackers("tracker")



    def train(self):
        print("start train")
        args = self.args
        accelerator = self.accelerator

        if accelerator.is_main_process and args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        def save_model_hook(trained_parameter_names, models, weights, output_dir):
            for i, model_to_save in enumerate(models):
                # model_to_save.save_pretrained(os.path.join(output_dir, "unet"), is_main_process=accelerator.is_main_process)
                save_directory = output_dir
                os.makedirs(save_directory, exist_ok=True)
                # Save the model
                # FSDP get state_dict need all the process
                _state_dict = model_to_save.state_dict()
                print(_state_dict)
                # requires_grad_keys = []
                # for key, para in model_to_save.named_parameters():
                #     if para.requires_grad:
                #         requires_grad_keys.append(key)
                state_dict = {}
                for key, value in _state_dict.items():
                    if key in trained_parameter_names:
                        state_dict[key] = value
                weights_name = 'model.bin'
                if accelerator.is_main_process:
                    torch.save(state_dict, os.path.join(save_directory, weights_name))
                    print(f"Model weights saved in {os.path.join(save_directory, weights_name)}")
            # make sure to pop weight so that corresponding model is not saved again
            if weights:
                weights.pop()
        save_model_hook = partial(save_model_hook, set(self.trained_parameter_names))
        accelerator.register_save_state_pre_hook(save_model_hook)
        # self.init_model()
        # self.init_for_train()
        global_step = 0
        train_steps = args.train_steps
        step = 0
        if args.eval_only:
            return
        progress_bar = ProgressBarTqdm(
            train_steps, disable=accelerator.is_local_main_process == False, smoothing=0)
        for _ in range(100):
            begin = time.perf_counter()
            batch: Any = next(self.train_dataiter)
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(self.model):
                # Convert images to latent space
                loss_dict = self.get_loss(batch)
                # Gather the losses across all processes for logging (if we use distributed training).
                loss_for_backward = sum(loss_dict.values())
                for key, value in loss_dict.items():
                    loss_dict[key] = accelerator.reduce(value, "mean").item()
                # Backward
                accelerator.backward(loss_for_backward)
                # grad = accelerator.clip_grad_norm_(model.trained_parameters, 1)
                self.optimizer.step()
                self.optimizer.zero_grad()

            accelerator.log(loss_dict, step=global_step)
            global_step += 1
            step += 1
            memory = torch.cuda.max_memory_allocated()
            memory_dict = dict(memory=f'{memory / 1e9:.3f}G')
            progress_bar.log({**loss_dict, **memory_dict}) # , "grad": grad.item()
            progress_bar.update()

            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
            if global_step >= train_steps:
                break
            if global_step % args.test_steps  == 0:
                self.inference(global_step=global_step)

        accelerator.save_state(save_path)
        self.inference(global_step=global_step)
        accelerator.end_training()

from torch.distributed.fsdp.wrap import _module_wrap_policy

from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel


def _test_fsdp_fp16(rank):
    from torch.optim import SGD
    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:12345", world_size=2, rank=rank
    )
    torch.cuda.set_device(rank)
    # rank = dist.get_rank()
    fsdp_model = FullyShardedDataParallel(
        module=Layer(), device_id=rank,
        auto_wrap_policy=partial(
            _module_wrap_policy,
            module_classes=[nn.Linear]))
    print(fsdp_model)
    print(list(fsdp_model.parameters())[0].numel())
    torch.distributed.barrier()
    optimizer = SGD(fsdp_model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(10000).cuda()
    for _ in range(10):
        optimizer.zero_grad()
        output = fsdp_model(data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        memory = torch.cuda.max_memory_allocated()
        if rank == 0:
            print(f'step memory allocate: {memory / 1e9:.3f}G')
        torch.cuda.reset_max_memory_allocated()

if __name__ == "__main__":
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Sequential(
                *(nn.Linear(10000, 10000) for _ in range(10))
            )

        def forward(self, x):
            return self.linear(x)


    def test_fp32():
        model = Layer().cuda()
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        data = torch.ones(10000).cuda()
        for i in range(10):
            optimizer.zero_grad()
            output = model(data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            memory = max_memory_allocated()
            print(f'step memory allocate: {memory / 1e9:.3f}G')


    def test_fp16():
        torch.cuda.init()
        model = Layer().cuda()
        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        data = torch.ones(10000).cuda()
        for _ in range(10):
            with autocast(device_type='cuda'):
                optimizer.zero_grad()
                output = model(data)
                loss = output.sum()
                loss.backward()
                optimizer.step()
            memory = max_memory_allocated()
            print(f'memory allocated: {memory / 1e9:.3f}G')

    FSDPTrain().train()
    # torch.multiprocessing.spawn(_test_fsdp_fp16, (), nprocs=2)
