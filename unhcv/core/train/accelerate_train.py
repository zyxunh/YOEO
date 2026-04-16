import itertools
import os
import argparse
import subprocess
import time
from functools import partial
from copy import deepcopy

import torch.distributed
from accelerate.state import AcceleratorState
from diffusers import EMAModel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import accelerate
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from typing import Any, Optional, Union, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import numpy as np
import wandb
from diffusers.optimization import get_scheduler as get_scheduler_, SchedulerType

from unhcv.common import CfgNode
from unhcv.common.image import pad_image_to_same_size
from unhcv.common.types import DataDict
from unhcv.common.utils import obj_dump, obj_load, write_txt, human_format_num, MeanCache, walk_all_files_with_suffix, \
    find_path, get_logger, ProgressBarTqdm, copy_file, ProgressBar, load_config
from unhcv.common.fileio import copy, remove_dir, listdir
from unhcv.common.utils.timer import TimerDict
from unhcv.core.utils import get_global_size
from unhcv.datasets.common_datasets import sharder_worker_init_fn, ShardWrapper, ShardWrapperTrain2Test
from unhcv.datasets.utils.test_data import get_train2test_iterabledataloader
from unhcv.nn.utils import filter_param, load_checkpoint, monitor_memory
from unhcv.nn.utils.dtype import DTYPE_MAP

from .configuration_train import TrainConfig


logger = get_logger(__name__)


def get_scheduler(*, name, optimizer, final_ratio, **kwargs):
    class IterExponential:
        def __init__(self, num_training_steps, final_ratio, num_warmup_steps=0) -> None:
            """
            Customized iteration-wise exponential scheduler.
            Re-calculate for every step, to reduce error accumulation

            Args:
                num_training_steps (int): Expected total iteration number
                final_ratio (float): Expected LR ratio at n_iter = total_iter_length
            """
            self.total_length = num_training_steps
            self.effective_length = num_training_steps - num_warmup_steps
            self.final_ratio = final_ratio
            self.warmup_steps = num_warmup_steps

        def __call__(self, n_iter) -> float:
            if n_iter < self.warmup_steps:
                alpha = 1.0 * n_iter / self.warmup_steps
            elif n_iter >= self.total_length:
                alpha = self.final_ratio
            else:
                actual_iter = n_iter - self.warmup_steps
                alpha = np.exp(
                    actual_iter / self.effective_length * np.log(self.final_ratio)
                )
            return alpha

    if name in {e.value for e in SchedulerType}:
        scheduler = get_scheduler_(name=name, optimizer=optimizer, **kwargs)
    else:
        if name == "iter_exponential":
            lr_func = IterExponential(final_ratio=final_ratio, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler name: {name}")
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)
    return scheduler


class AccelerateTrainer:

    frozen_models: Optional[List[nn.Module]] = []
    extra_models: Optional[List[nn.Module]] = []
    accelerator: Optional[Accelerator] = None
    state_dict_save_path_queue: Dict[str, List[str]] = dict(local=[], hdfs=[])
    log_cache_memory = {}
    last_save_state_step = -1
    mean_cache = MeanCache()
    proc_asynchronization = []
    global_step: int = 0
    save_for_training_show_tensors: Optional[DataDict] = None
    inference_dataloader: Optional[DataLoader] = None
    test_dataloader: Optional[Tuple[DataLoader]] = None
    prepare_train_dataloader = False

    progress_bar: ProgressBar = None

    def __init__(self, config: TrainConfig=TrainConfig(), *args, **kwargs) -> None:
        self.config = config

        self.model: Optional[Union[nn.Module, Tuple[nn.Module]]] = None
        self.logger = None
        self.args: Optional[argparse.Namespace] = None
        self.train_dataiter: Optional[iter] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.weight_dtype: Optional[torch.dtype] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.demo_dataset: Optional[Dataset] = None
        self.demo_dataloader: Optional[DataLoader] = None

        self.init_log()
        self.get_args()
        self.accelerator = self.init_accelerator()
        self.logger.info(self.args)

        self.args.debug = self.args.debug or self.args.debug_mode is not None
        if (not self.args.eval_only) and self.args.debug_mode != "model" and self.args.debug_mode != "dataset":
            self.init_train_dataset()
        if self.args.debug_mode == "dataset":
            self.debug_for_dataset()
            self.accelerator.wait_for_everyone()
            exit()
        elif self.args.debug_mode == "demo":
            self.init_for_demo()
            self.debug_for_demo()
            exit()
        else:
            pass

        self.init_model()
        self.froze_model()

        if self.args.checkpoint is not None:
            checkpoint = self.args.checkpoint
            if os.path.isfile(checkpoint):
                checkpoints = (checkpoint, )
            else:
                checkpoints = walk_all_files_with_suffix(checkpoint, ".bin")
            models = self.model
            for model, checkpoint in zip(models, checkpoints):
                load_checkpoint(model, find_path(checkpoint), log_missing_keys=True)

        # self.trained_parameter_names = trained_parameter_names = []
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         trained_parameter_names.append(name)
        # self.build_test_dataloader()
        if self.args.eval_only:
            self.init_for_eval()
        else:
            self.init_for_train()
        self.save_project_information()
        if self.args.debug_mode == "model":
            exit()

    def parser_add_argument(self):
        parser = argparse.ArgumentParser(description="Simple example of a training script.")
        parser.add_argument(
            "--checkpoint_root",
            type=str,
            default="/home/tiger/train_outputs/checkpoint"
        )
        parser.add_argument(
            "--show_root",
            type=str,
            default="/home/tiger/train_outputs/show"
        )
        parser.add_argument(
            "--hdfs_root",
            type=str,
            default=None
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--show_dir",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--hdfs_dir",
            type=str,
            default=None,
        )
        parser.add_argument("--max_local_state_num", type=int, default=1)
        parser.add_argument("--max_hdfs_state_num", type=int, default=3)
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
            help="Learning rate to use.",
        )
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="constant",
            help=(
                'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                ' "constant", "constant_with_warmup"]'
            ),
        )
        parser.add_argument(
            "--lr_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler."
        )
        parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        parser.add_argument("--max_grad_norm", type=float, default=0)

        parser.add_argument("--num_train_epochs", type=int, default=100)
        parser.add_argument(
            "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
        )
        parser.add_argument("--effective_batch_size", type=int, default=None)
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument(
            "--dataloader_num_workers",
            type=int,
            default=0,
            help=(
                "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
            ),
        )
        parser.add_argument(
            "--save_steps",
            type=int,
            default=2000,
            help=(
                "Save a checkpoint of the training state every X updates"
            ),
        )
        parser.add_argument(
            "--test_steps",
            type=int,
            default=1000
        )
        parser.add_argument(
            "--train_steps",
            type=int,
            default=100000
        )
        parser.add_argument(
            "--train_visual_steps",
            type=int,
            default=None
        )
        parser.add_argument(
            "--mixed_precision",
            type=str,
            default=None,
            choices=["no", "fp16", "bf16"],
            help=(
                "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
            ),
        )
        parser.add_argument(
            "--report_to",
            type=str,
            default="tensorboard",
            help=(
                'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
            ),
        )
        parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        parser.add_argument("--extra_checkpoint", type=str, default=None)
        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--eval_only", action="store_true")
        parser.add_argument("--eval_before_train", action="store_true")
        parser.add_argument("--not_eval_training", action="store_true")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--debug_mode", type=str, default=None, choices=("dataset", "model", "demo"))
        parser.add_argument("--project_name", type=str, default=None)
        parser.add_argument("--project_suffix", type=str, default=None)
        parser.add_argument("--dataset_config", type=str, default=None, nargs='+')
        parser.add_argument("--model_config", type=str, default=None, nargs='+')
        parser.add_argument("--args_file", type=str, default=None)
        parser.add_argument("--demo_dir", type=str, default=None)
        parser.add_argument("--save_nongrad", action='store_true')
        parser.add_argument("--save_dtype", type=str, default=None)
        parser.add_argument("--visual", action='store_true')
        parser.add_argument("--metric_collect_file", type=str, default=None)
        parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                            help="Whether or not to use xformers.")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--seed_hard", action='store_true')
        parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
        parser.add_argument("--ema_update_after_step",  type=int, default=0)
        return parser

    def get_args(self):
        parser = self.parser_add_argument()
        args = parser.parse_args()
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if args.args_file is not None:
            args_loaded = obj_load(args.args_file)["updated_args"]
            for key in args.__dict__.keys():
                if getattr(args, key) == parser.get_default(key):
                    setattr(args, key, getattr(args_loaded, key))
        if args.project_name is None:
            if args.model_config is not None:
                name = ','.join([os.path.basename(os.path.splitext(var)[0]) for var in args.model_config])
                args.project_name = "model-{}".format(name)
            else:
                args.project_name = "model-null"
            if args.dataset_config is not None:
                name = ','.join([os.path.basename(os.path.splitext(var)[0]) for var in args.dataset_config])
                args.project_name += "_dataset-{}".format(name)
        if args.project_suffix is not None:
            args.project_name += "_" + args.project_suffix
        if args.checkpoint_dir is None:
            args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
        if args.show_dir is None:
            args.show_dir = os.path.join(args.show_root, args.project_name)
        if args.hdfs_dir is None and args.hdfs_root is not None:
            args.hdfs_dir = os.path.join(args.hdfs_root, args.project_name)
        if args.train_visual_steps is None:
            args.train_visual_steps = args.test_steps
        if args.debug:
            args.checkpoint_dir += "_debug"
            args.show_dir += "_debug"
        if args.eval_only:
            args.checkpoint_dir += "_eval"
            args.show_dir += "_eval"
        if env_local_rank == -1 or env_local_rank == 0:
            if args.debug:
                remove_dir(args.checkpoint_dir)
        if env_local_rank != -1:
            args.local_rank = env_local_rank
        self.args = args

        # region merge config文件中的训练设置
        model_config = self.parse_model_config()
        if "scheduler_config" in model_config:
            self.config.scheduler_config.merge(model_config.get("scheduler_config"))
        if "optim_config" in model_config:
            self.config.optim_config.merge(model_config.get("optim_config"))
        if "train_config" in model_config:
            self.config.merge(model_config.get("train_config"))
        if self.config.train_steps is not None:
            args.train_steps = self.config.train_steps
        if self.config.train_batch_size is not None:
            args.train_batch_size = self.config.train_batch_size
        if self.config.effective_batch_size is not None:
            args.effective_batch_size = self.config.effective_batch_size
        if args.effective_batch_size is not None:
            args.gradient_accumulation_steps = args.effective_batch_size // args.train_batch_size // get_global_size()
        if self.config.optim_config.learning_rate is not None:
            args.learning_rate = self.config.optim_config.learning_rate
        if self.config.scheduler_config.name is not None:
            args.lr_scheduler = self.config.scheduler_config.name
        # endregion

        if env_local_rank == -1 or env_local_rank == 0:
            default_and_updated_args = dict(default_args={}, updated_args={})
            for key, value in args.__dict__.items():
                if value == parser.get_default(key):
                    default_and_updated_args["default_args"][key] = value
                else:
                    default_and_updated_args["updated_args"][key] = value
            obj_dump(os.path.join(args.checkpoint_dir, 'information', 'args.yml'),
                     default_and_updated_args)
            command = (f"tar -czvf {os.path.join(args.checkpoint_dir, 'information', 'code.tar')} \
                      --exclude='*.pyc' \
                      --exclude='__pycache__' \
                      --exclude='.git' \
                      --exclude='venv' \
                      --exclude='*.log' --exclude='*.jpg' --exclude='*.png' --exclude='*.egg-info'\
                      -C {find_path('code')} {'unhcv_refactor'}")
            subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)

    def init_log(self):
        self.logger = get_logger(__name__, accelerate_mode=True)

    def init_model(self):
        accelerator = self.accelerator
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

    def froze_model(self):
        for model in self.frozen_models:
            if model is not None:
                model.requires_grad_(False)

    @property
    def trained_parameters(self):
        return tuple([filter_param(var.parameters()) for var in self.model])

    def parse_model_config(self):
        config = load_config(self.args.model_config) if self.args.model_config else {}
        if self.accelerator and self.accelerator.is_main_process and self.args.model_config is not None:
            tmp = ','.join([os.path.basename(os.path.splitext(var)[0]) for var in self.args.model_config])
            suffix = os.path.splitext(self.args.model_config[0])[1]
            config.dump(os.path.join(self.args.checkpoint_dir, 'information', 'model-{}{}'.format(tmp, suffix)))
        return config

    def parse_dataset_config(self)-> Dict:
        config = load_config(self.args.dataset_config) if self.args.dataset_config else {}
        if self.accelerator.is_main_process and self.args.dataset_config is not None:
            tmp = ','.join([os.path.basename(os.path.splitext(var)[0]) for var in self.args.dataset_config])
            suffix = os.path.splitext(self.args.dataset_config[0])[1]
            config.dump(os.path.join(self.args.checkpoint_dir, 'information', 'dataset-{}{}'.format(tmp, suffix)))
        return config

    def parse_train_dataset_config(self):
        return self.parse_dataset_config()

    def parse_test_dataset_config(self):
        return self.parse_dataset_config()

    @staticmethod
    def build_dataset(dataset_class, default_dataset_kwargs: CfgNode, dataset_config, batch_size):
        if not isinstance(default_dataset_kwargs, tuple):
            default_dataset_kwargs = (default_dataset_kwargs,)
        else:
            default_dataset_kwargs = default_dataset_kwargs
        train_datasets = []
        for dataset_kwargs in default_dataset_kwargs:
            dataset_kwargs = deepcopy(dataset_kwargs)
            dataset_kwargs.merge(dataset_config)
            dataset_kwargs["batch_size"] = batch_size
            train_dataset: Dataset = dataset_class(dataset_kwargs)
            train_datasets.append(train_dataset)
        if len(train_datasets) > 1:
            train_dataset = ConcatDataset(train_datasets)
        else:
            train_dataset = train_datasets[0]
        return train_dataset

    def init_train_dataset(self):
        args = self.args
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        dataset_config = self.parse_train_dataset_config()
        train_dataset = self.build_dataset(self.config.dataset_class, self.config.default_dataset_kwargs, dataset_config, args.train_batch_size)
        train_dataset = ShardWrapper(train_dataset, random_shuffle=True, drop_last=False, infinite=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size,
                                      num_workers=args.dataloader_num_workers,
                                      prefetch_factor=8 if args.dataloader_num_workers > 0 else None,
                                      collate_fn=self.config.dataset_collate_fn, pin_memory=True, shuffle=False,
                                      drop_last=False, worker_init_fn=sharder_worker_init_fn)

        self.logger.info("***** Running training *****")
        self.logger.info(
            f"world_size is {world_size}, batch_size is {world_size * args.train_batch_size * args.gradient_accumulation_steps}, "
            f"dataset num is {len(train_dataset)}, "
            f"num_epoch is {world_size * args.train_batch_size * args.train_steps * args.gradient_accumulation_steps / len(train_dataset)}")
        time.sleep(10)
        self.train_dataloader = train_dataloader
        return train_dataloader

    def init_test_dataset(self, test_for_which="train", test_num=50, dataset_config=None, max_data_num=None):
        if test_for_which == "train":
            if dataset_config is None:
                dataset_config = self.parse_train_dataset_config()
            test_dataset = self.build_dataset(self.config.test_dataset_class, self.config.default_test_dataset_kwargs,
                                               dataset_config, 1)
            test_dataset = ShardWrapperTrain2Test(test_dataset, random_shuffle=False, drop_last=False, infinite=False, num=max_data_num)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=1,
                                         num_workers=1, #self.args.dataloader_num_workers,
                                         prefetch_factor=2,
                                         collate_fn=self.config.dataset_collate_fn, pin_memory=True, shuffle=False,
                                         drop_last=False, worker_init_fn=sharder_worker_init_fn)
            return test_dataloader
        elif test_for_which == "test":
            if dataset_config is None:
                dataset_config = self.parse_test_dataset_config()
            test_dataset = self.build_dataset(self.config.test_dataset_class, self.config.default_test_dataset_kwargs,
                                              dataset_config, 1)
            test_dataset = ShardWrapper(test_dataset, random_shuffle=False, drop_last=False, infinite=False)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=1,
                                         num_workers=1, #self.args.dataloader_num_workers,
                                         prefetch_factor=2,
                                         collate_fn=self.config.dataset_collate_fn, pin_memory=True, shuffle=False,
                                         drop_last=False, worker_init_fn=sharder_worker_init_fn)
            return test_dataloader
        else:
            raise NotImplementedError("test_for_which should be train or test")

    def prepare(self, *args, **kwargs):
        _args = ()
        for x in args:
            if isinstance(x, tuple):
                _args += x
            elif x is None:
                pass
            else:
                _args += (x,)
        _args = self.accelerator.prepare(*_args, **kwargs)
        if not isinstance(_args, tuple):
            _args = (_args,)
        out = []
        for x in args:
            if isinstance(x, tuple):
                out += (_args[:len(x)],); _args = _args[len(x):]
            elif x is None:
                out += (None,)
            else:
                out += (_args[0],); _args = _args[1:]
        return out

    def save_model_hook(self, models, weights, output_dir, filter_nongrad=True, dtype=None, model_name="model"):
        for i, model_to_save in enumerate(models):
            # model_to_save.save_pretrained(os.path.join(output_dir, "unet"), is_main_process=accelerator.is_main_process)
            save_directory = output_dir
            os.makedirs(save_directory, exist_ok=True)
            # Save the model
            # FSDP get state_dict need all the process
            _state_dict: Dict[str, Any] = model_to_save.state_dict()
            if dtype is not None:
                for k, v in _state_dict.items():
                    _state_dict[k] = v.to(dtype=dtype)
            not_requires_grad_keys = []
            state_dict_not_parameters = []
            if filter_nongrad:
                requires_grad_keys = []
                for key, para in model_to_save.named_parameters():
                    if para.requires_grad:
                        requires_grad_keys.append(key)
                    else:
                        not_requires_grad_keys.append(key)
                requires_grad_keys = set(requires_grad_keys)
            else:
                requires_grad_keys = None
            self.logger.info(f"not requires grad keys: {not_requires_grad_keys}")
            state_dict = {}
            for key, value in _state_dict.items():
                if requires_grad_keys is None or key in requires_grad_keys or key.endswith("running_mean") or key.endswith("running_var") or key.endswith("num_batches_tracked"):
                    if key.startswith("module."):
                        key = key[len("module."):]
                    state_dict[key] = value
                else:
                    state_dict_not_parameters.append(key)
            self.logger.info(f"state dict but not parameters keys: {state_dict_not_parameters}")
            weights_name = f'{model_name}_{i}.bin'
            if self.accelerator.is_main_process:
                torch.save(state_dict, os.path.join(save_directory, weights_name))
                self.logger.info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")
        # make sure to pop weight so that corresponding model is not saved again
        if weights:
            weights.pop()

    def init_ema(self):
        if self.args.use_ema:
            ema_model = [deepcopy(var) for var in self.model]
            self.ema_model = ema_model = tuple([EMAModel(var.parameters()) for var in ema_model])
            _ = [var.to(self.accelerator.device) for var in ema_model]

    def init_for_train(self):
        args = self.args
        accelerator = self.accelerator
        model = self.model
        trained_parameters = self.trained_parameters
        self.init_ema()
        optimizer = ()
        optim_cls = getattr(torch.optim, self.config.optim_config.name)
        for i_optim in range(len(trained_parameters)):
            optimizer += (
            optim_cls(trained_parameters[i_optim], lr=args.learning_rate, weight_decay=args.weight_decay),)
        # Prepare everything with our `accelerator`.
        lr_scheduler = ()
        for i_optim in range(len(optimizer)):
            lr_scheduler += (get_scheduler(name=args.lr_scheduler, optimizer=optimizer[i_optim],
                                           num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                           num_training_steps=args.train_steps * accelerator.num_processes, final_ratio=self.config.scheduler_config.final_ratio),)

        if self.prepare_train_dataloader:
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.prepare(model, optimizer, self.train_dataloader, lr_scheduler)
        else:
            self.model, self.optimizer, self.lr_scheduler = self.prepare(model, optimizer, lr_scheduler)

        # train_dataiter = iter(self.train_dataloader)
        # self.train_dataiter = train_dataiter
        save_model_hook = partial(self.save_model_hook, filter_nongrad=not args.save_nongrad)
        accelerator.register_save_state_pre_hook(save_model_hook)

    def init_for_eval(self):
        self.model = tuple([model.to(device="cuda", dtype=self.weight_dtype) for model in self.model])
        self.init_ema()
        # self.model = self.prepare(self.model)[0]
        # return self.init_for_demo()

    @staticmethod
    def build_naive_dataset(dataset_cls, dataset_config, batch_size):
        dataset = dataset_cls(**dataset_config)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=torch.utils.data.default_collate,
            pin_memory=True,
            shuffle=False)
        return dataloader

    def init_for_demo(self):
        if self.config.default_demo_dataset_kwargs is not None:
            default_demo_dataset_kwargs = self.config.default_demo_dataset_kwargs.copy()
            default_demo_dataset_kwargs["batch_size"] = self.config.demo_batch_size
            dataset_config = self.parse_dataset_config()
            default_demo_dataset_kwargs.update(dataset_config.pop("demo_dataset_kwargs", {}))
            self.demo_dataloader = self.build_naive_dataset(self.config.demo_dataset_class, default_demo_dataset_kwargs, self.config.demo_batch_size)
            return True
        return False

    def backward(self, loss_dict):
        losses = []
        for key, value in loss_dict.items():
            if value.requires_grad:
                losses.append(value)
            loss_dict[key] = value.detach()
        losses = sum(losses)
        self.accelerator.backward(losses)
        return loss_dict

    def get_loss(self, batch: Any):
        if self.global_step % self.args.train_visual_steps == 1:
            self.save_for_training_show_tensors = DataDict()
        loss = dict()
        return loss

    def inference_on_train(self, global_step):
        pass

    def build_test_dataloader(self):
        dataset_config = self.parse_dataset_config()
        args = self.args
        if self.config.test_dataset_class is None:
            return
        if isinstance(self.config.default_test_dataset_kwargs, dict):
            default_dataset_kwargs = (self.config.default_test_dataset_kwargs,)
        else:
            default_dataset_kwargs = self.config.default_test_dataset_kwargs
        test_dataloaders = ()
        for dataset_kwargs in default_dataset_kwargs:
            dataset_kwargs = dataset_kwargs.copy()
            dataset_kwargs.update(dataset_config.get("test_dataset_kwargs", {}))
            dataset_kwargs["batch_size"] = 1
            test_dataset: Dataset = self.config.test_dataset_class(dataset_kwargs)
            test_dataset = ShardWrapper(test_dataset, random_shuffle=False, drop_last=False, infinite=False)
            test_dataloaders += (DataLoader(dataset=test_dataset, batch_size=1, num_workers=args.dataloader_num_workers,
                                            prefetch_factor=2,
                                            collate_fn=torch.utils.data.default_collate, pin_memory=True, shuffle=False,
                                            drop_last=False, worker_init_fn=sharder_worker_init_fn),)
        self.test_dataloader = test_dataloaders

    def inference_on_test(self, global_step):
        show_root = os.path.join(self.args.show_dir, 'test/{:06}'.format(self.global_step))
        test_dataloaders = self.init_test_dataset(test_for_which="test")
        if self.args.use_ema:
            for ema_model_, model_ in zip(self.ema_model, self.model):
                ema_model_.store(model_.parameters())
                ema_model_.copy_to(model_.parameters())
        model = self.accelerator.unwrap_model(self.model[0]).eval()
        if not isinstance(test_dataloaders, tuple):
            test_dataloaders = (test_dataloaders,)
        for test_dataloader in test_dataloaders:
            for data in test_dataloader:
                with torch.no_grad():
                    out = model(data)
        if self.args.use_ema:
            for ema_model_, model_ in zip(self.ema_model, self.model):
                ema_model_.restore(model_.parameters())

    def inference_on_demo(self, global_step):
        show_root = os.path.join(self.args.show_dir, 'show_demo/{:06}'.format(global_step))
        pass

    def visual_training_result(self):
        show_root = os.path.join(self.args.show_dir, 'training/{:06}'.format(self.global_step))
        save_for_training_show_tensors = self.save_for_training_show_tensors
        # self.args.max_local_state_num
        self.config.max_visual_num_in_training
        pass

    @property
    def is_visual_iter(self):
        return self.progress_bar.is_visual_iter and self.accelerator.is_main_process

    def log_images_to_cache(self, images, tag="image", captions=None, reverse_color=False):
        if not isinstance(images, Sequence):
            images = [images]
        if reverse_color:
            images = [var[..., ::-1] for var in images]
        assert captions is None
        memory = self.log_cache_memory.get(tag, [])
        memory.extend(images)
        self.log_cache_memory[tag] = memory

    def log_images_cache_push(self):
        for tag, images in self.log_cache_memory.items():
            self.log_images(images, tag=tag)
        self.log_cache_memory.clear()

    def log_images(self, images, tag="image", captions=None, reverse_color=False):
        if not isinstance(images, Sequence):
            images = [images]
        images = pad_image_to_same_size(images)
        if reverse_color:
            images = [var[..., ::-1] for var in images]
        if captions is None:
            captions = ["{:3}".format(i) for i in range(len(images))]
        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images: np.ndarray = np.stack([np.asarray(img) for img in images])
                if np_images.ndim == 3:
                    np_images = np_images[..., None]
                tracker.writer.add_images(tag, np_images, self.global_step, dataformats="NHWC")
            elif tracker.name == "wandb":
                tracker.log(
                    {
                        tag: [
                            wandb.Image(image, caption=f"{captions[i]}")
                            for i, image in enumerate(images)
                        ]
                    }, step=self.global_step
                )
            else:
                self.logger.warning(f"image logging not implemented for {tracker.name}")

    def _debug_for_visual_dataset(self, dataset_iter, mode, suffix=None):
        for batch_index in range(100):
            data = next(dataset_iter)
            if mode == 'train':
                self.visual_train_dataset(data, batch_index)
            elif mode == 'test':
                self.visual_test_dataset(data, batch_index, suffix)
            print("data index is {:05}".format(batch_index))

    def _debug_for_dataset(self, mode='train'):
        if mode == 'train':
            self.init_train_dataset()
            dataset_iter = iter(self.train_dataloader.dataset)
            if hasattr(self.train_dataloader.dataset, 'debug'):
                self.train_dataloader.dataset.debug()
            self._debug_for_visual_dataset(dataset_iter, mode)
        elif mode == 'test':
            dataloaders = self.init_test_dataset(test_for_which="test")
            for i_dataloader, dataloader in enumerate(dataloaders):
                if hasattr(dataloader.dataset, 'debug'):
                    dataloader.dataset.debug()
                dataset_iter = iter(dataloader.dataset)
                self._debug_for_visual_dataset(dataset_iter, mode, suffix=f"{i_dataloader}")

        if mode == 'train':
            dataiter = iter(self.train_dataloader)

        for batch_index in range(100):
            data = next(dataiter)
            if mode == 'train':
                self.visual_train_dataloader(data, batch_index)
            elif mode == 'test':
                self.visual_test_dataloader(data, batch_index)
            print("data index is {:05}".format(batch_index))

    def debug_for_dataset(self):
        self._debug_for_dataset(mode='train')
        self._debug_for_dataset(mode='test')

    def visual_train_dataset(self, data, batch_index):
        pass

    def visual_train_dataloader(self, data, batch_index):
        pass

    def visual_test_dataset(self, data, batch_index):
        pass

    def visual_test_dataloader(self, data, batch_index):
        pass

    def debug_for_demo(self):
        show_root = os.path.join(self.args.show_dir, 'demo_dataset')
        remove_dir(show_root)
        for i_data, data in enumerate(self.demo_dataloader):
            pass

    def inference(self, global_step):
        if self.demo_dataloader is not None:
            self.inference_on_demo(global_step=global_step)
            self.accelerator.wait_for_everyone()
        # self.build_test_dataloader()
        self.inference_on_test(global_step=global_step)
        self.test_dataloader = None
        self.accelerator.wait_for_everyone()
        self.inference_on_train(global_step=global_step)
        self.accelerator.wait_for_everyone()
        import gc; gc.collect()

    def init_accelerator(self):
        args = self.args
        logging_dir = os.path.join(args.show_dir, "log")
        accelerator_project_config = ProjectConfiguration(project_dir=args.checkpoint_dir, logging_dir=logging_dir)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )

        if args.seed is not None:
            logger.info("set seed to {}".format(args.seed))
            set_seed(args.seed)
            if args.seed_hard:
                logger.info("set seed use_deterministic_algorithms")
                torch.use_deterministic_algorithms(True, warn_only=True)

        # accelerator.deepspeed_engine_wrapped
        if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "gradient_accumulation_steps"] = args.gradient_accumulation_steps
            accelerator.state.deepspeed_plugin.deepspeed_config[
                'train_micro_batch_size_per_gpu'] = args.train_batch_size

        if accelerator.is_main_process:
            if args.report_to == "tensorboard" and not args.eval_only:
                remove_dir(logging_dir)
            os.makedirs(logging_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = logging_dir
            accelerator.init_trackers(args.project_name, config={k: v for k, v in dict(vars(args)).items() if not isinstance(v, list)},
                                      init_kwargs={"wandb": {"mode": "offline"}, "tensorboard": {}})
        return accelerator

    @staticmethod
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = accelerate.state.AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    @staticmethod
    def save_model_information(model, mode_name=None, save_dir=None, hdfs_dir=None, log_only=False):
        if model is None:
            return
        if not log_only:
            write_txt(
                os.path.join(save_dir, 'information', f'{mode_name}.txt'),
                model.__str__())
        trained_parameters = []
        trained_parameters_num = 0
        frozen_parameters_num = 0
        frozen_parameters = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trained_parameters_num += param.numel()
                trained_parameters.append(f"{name}: {list(param.shape)}\n")
            else:
                frozen_parameters_num += param.numel()
                frozen_parameters.append(f"{name}: {list(param.shape)}\n")

        if log_only:
            print(f"trained_parameters_num: {human_format_num(trained_parameters_num)}\n"
                f"frozen_parameters_num: {human_format_num(frozen_parameters_num)}\n"
                f"all_parameters_num: {human_format_num(trained_parameters_num + frozen_parameters_num)}")
            print(f"trained_parameters_num: {trained_parameters_num}\n"
                f"frozen_parameters_num: {frozen_parameters_num}\n"
                f"all_parameters_num: {trained_parameters_num + frozen_parameters_num}")
        else:
            write_txt(
                os.path.join(save_dir, 'information',
                             f'{mode_name}_trained_parameters.txt'), trained_parameters)
            write_txt(
                os.path.join(save_dir, 'information',
                             f'{mode_name}_frozen_parameters.txt'), frozen_parameters)
            write_txt(
                os.path.join(save_dir, 'information',
                             f'{mode_name}_parameters_num.txt'),
                f"trained_parameters_num: {human_format_num(trained_parameters_num)}\n"
                f"frozen_parameters_num: {human_format_num(frozen_parameters_num)}\n"
                f"all_parameters_num: {human_format_num(trained_parameters_num + frozen_parameters_num)}"
            )
        if hdfs_dir is not None:
            for file in listdir(save_dir):
                copy(file, hdfs_dir)

    @staticmethod
    def analyse_optim(optim, name, save_dir, hdfs_dir=None):
        if optim is None:
            return
        param_groups = optim.param_groups
        information = []
        for i_group, param_group in enumerate(param_groups):
            param_shapes = []
            information.append(f"param_group: {i_group}\n")
            for key, value in param_group.items():
                if key == "params":
                    continue
                information.append(f"{key}: {value}\n")
            params = param_group["params"]
            params_num = 0
            for param in params:
                params_num += param.numel()
                param_shapes.append(f"{str(list(param.shape))}\n")
            information.append(f"params_num: {human_format_num(params_num)}\n")
            information.extend([f"\n", *param_shapes])
        write_txt(os.path.join(save_dir, 'information', f'{name}.txt'), information)

    def save_project_information(self):
        if self.accelerator.is_main_process:
            models = self.model
            for i_model, model in enumerate(models):
                self.save_model_information(model, f"train_model_{i_model + 1}", self.args.checkpoint_dir, self.args.hdfs_dir)
            for i_model, model in enumerate(self.frozen_models):
                self.save_model_information(model, f"model_{i_model + 1}", self.args.checkpoint_dir, self.args.hdfs_dir)
            for i_model, model in enumerate(self.extra_models):
                self.save_model_information(model, f"extra_model_{i_model + 1}", self.args.checkpoint_dir,
                                            self.args.hdfs_dir)
            if self.optimizer is not None:
                if isinstance(self.optimizer, torch.optim.Optimizer):
                    optimizer = [self.optimizer]
                else:
                    optimizer = self.optimizer
                for i_optim, optim in enumerate(optimizer):
                    self.analyse_optim(optim, f"optim_{i_optim}", self.args.checkpoint_dir, self.args.hdfs_dir)

        self.accelerator.wait_for_everyone()

    def save_state(self):
        if self.last_save_state_step != self.global_step:
            self.last_save_state_step = self.global_step
        else:
            return
        save_path = os.path.join(self.args.checkpoint_dir, f"checkpoint-{self.global_step}")
        # self.accelerator.save_state(save_path)
        if self.accelerator.is_main_process:
            model = [self.accelerator.unwrap_model(var) for var in self.model]
            self.save_model_hook(model, None, output_dir=save_path, filter_nongrad=not self.args.save_nongrad, dtype=DTYPE_MAP[self.args.save_dtype])
            if self.args.use_ema:
                for i_model, ema_model in enumerate(self.ema_model):
                    torch.save(ema_model.state_dict(), os.path.join(save_path, f"model_{i_model}_ema.pth"))
            del model
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            files = listdir(save_path)
            for file in files:
                if "model" not in os.path.basename(file):
                    remove_dir(file)
            if self.args.max_local_state_num is not None:
                self.state_dict_save_path_queue['local'].append(save_path)
                if len(self.state_dict_save_path_queue['local']) > self.args.max_local_state_num:
                    remove_dir(self.state_dict_save_path_queue['local'][0])
                    del self.state_dict_save_path_queue['local'][0]
            if self.args.hdfs_dir is not None:
                hdfs_save_path = os.path.join(self.args.hdfs_dir, f"checkpoint-{self.global_step}")
                self.proc_asynchronization.append(copy(save_path, hdfs_save_path, asynchronization=True))
                if self.args.max_hdfs_state_num is not None:
                    self.state_dict_save_path_queue['hdfs'].append(hdfs_save_path)
                    if len(self.state_dict_save_path_queue['hdfs']) > self.args.max_hdfs_state_num:
                        remove_dir(self.state_dict_save_path_queue['hdfs'][0])
                        del self.state_dict_save_path_queue['hdfs'][0]
        self.accelerator.wait_for_everyone()

    def main(self):
        if self.args.eval_only:
            self.inference(global_step=0)
        else:
            if self.args.debug:
                self.logger.info("verify save_state")
                self.save_state()
                self.logger.info("verify inference")
                self.inference(global_step=0)
            self.train()

    def train_step(self, batch):
        accelerator = self.accelerator
        args = self.args
        loss_dict = self.get_loss(batch)
        # Gather the losses across all processes for logging (if we use distributed training).
        loss_for_backward = sum(loss_dict.values())
        for key, value in loss_dict.items():
            loss_dict[key] = accelerator.reduce(value, "mean").item()
        loss_sum = accelerator.reduce(loss_for_backward, "mean").item()
        self.mean_cache.update(dict(loss=loss_sum))
        self.mean_cache.update(loss_dict)
        # Backward
        if self.args.debug:
            monitor_memory("before loss_for_backward")
        accelerator.backward(loss_for_backward)
        if self.args.debug:
            monitor_memory("after loss_for_backward")
        if args.max_grad_norm != 0 and accelerator.sync_gradients:
            models = self.model
            grad_before_norm = accelerator.clip_grad_norm_(itertools.chain(*[model.parameters() for model in models]), args.max_grad_norm)
            if grad_before_norm is not None:
                if isinstance(grad_before_norm, torch.Tensor):
                    grad_before_norm = grad_before_norm.item()
                self.mean_cache.update({f"grad": grad_before_norm})

        if self.args.debug:
            monitor_memory("before optimizer")
        for optimizer in self.optimizer:
            optimizer.step()
        for lr_scheduler in self.lr_scheduler:
            lr_scheduler.step()
        for optimizer in self.optimizer:
            optimizer.zero_grad()
        if self.args.debug:
            monitor_memory("after optimizer")

    def train(self):
        print("start train")
        args = self.args
        accelerator = self.accelerator
        time_dict = TimerDict()

        if accelerator.is_main_process and args.checkpoint_dir is not None:
            os.makedirs(args.checkpoint_dir, exist_ok=True)


        train_steps = args.train_steps
        if args.eval_only:
            return
        self.progress_bar = progress_bar = ProgressBar(train_steps, disable=not accelerator.is_local_main_process,
                                                       smoothing=0.1,
                                                       file=os.path.join(args.show_dir, "log", "log.txt"), name=self.args.project_name)
        if args.eval_before_train:
            self.inference(global_step=self.global_step)
        if self.train_dataiter is None:
            self.train_dataiter = iter(self.train_dataloader)
        while True:
            time_dict.tic("data")
            try:
                batch: Any = next(self.train_dataiter)
            except StopIteration:
                self.train_dataiter = iter(self.train_dataloader)
                batch: Any = next(self.train_dataiter)
            time_dict.toc("data")
            time_dict.tic("model")
            with accelerator.accumulate(*self.model):
                # Convert images to latent space
                self.train_step(batch)
            time_dict.toc("model")

            time_dict.tic("other")
            if accelerator.sync_gradients:
                self.ema_step()
                self.global_step += 1
                tacker_logs = {}
                mean_dict = self.mean_cache.mean()
                tacker_logs.update(mean_dict)
                lr_scheduler = self.lr_scheduler
                for j_lr, _lr_scheduler in enumerate(lr_scheduler):
                    for i_lr, lr in enumerate(_lr_scheduler.get_lr()):
                        tacker_logs[f"lr_{j_lr}_{i_lr}"] = lr
                accelerator.log(tacker_logs, step=self.global_step)
                progress_bar_logs = mean_dict.copy() # dict(loss=mean_dict.get('loss'))
                grad = mean_dict.get("grad", None)
                if grad is not None:
                    progress_bar_logs["grad"] = grad
                progress_bar_logs.update(time_dict.get_mean_time())

                progress_bar.log(progress_bar_logs)  # , "grad": grad.item()
                progress_bar.update()

                if self.global_step % args.save_steps == 0:
                    self.save_state()
                if self.global_step >= train_steps:
                    break
                if self.global_step % args.test_steps == 0:
                    self.inference(global_step=self.global_step)
                if self.save_for_training_show_tensors is not None:
                    if self.accelerator.is_main_process:
                        self.visual_training_result()
                    self.save_for_training_show_tensors = None
                self.accelerator.wait_for_everyone()
            time_dict.toc("other")

        # save_path = os.path.join(args.checkpoint_dir, f"checkpoint-{self.global_step}")
        self.save_state()
        # if not os.path.exists(save_path):
        #     accelerator.save_state(save_path)
        if not self.args.not_eval_training:
            self.inference(global_step=self.global_step)
        # accelerator.end_training()
        self.end_training()

    def ema_step(self):
        if self.args.use_ema:
            for ema_model_, model_ in zip(self.ema_model, self.model):
                ema_model_.step(model_.parameters())

    def end_training(self):
        for proc in self.proc_asynchronization:
            raise NotImplementedError
            stdout, stderr = proc.communicate()
            if stdout is not None or stderr is not None:
                print(f'stdout is {stdout}, strderr is {stderr}')
        self.accelerator.end_training()
        os.rename(self.args.checkpoint_dir, self.args.checkpoint_dir + "_done")
        os.rename(self.args.show_dir, self.args.show_dir + "_done")

    def convert_model(self, train=False):
        models = self.model
        if train:
            models = [var.train() for var in models]
        else:
            models = [self.accelerator.unwrap_model(var).eval() for var in models]
        return list(models)

    def visual(self, *args, **kwargs):
        raise NotImplementedError


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


if __name__ == "__main__":
    pass
