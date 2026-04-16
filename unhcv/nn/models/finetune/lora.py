import torch
import torch.nn as nn
import math
from peft.tuners.tuners_utils import BaseTunerLayer

class LoraConv2d(BaseTunerLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, r=4, lora_alpha=1.0, adapter_name="default"):
        super().__init__()

        # 主卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_flag = bias

        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()

        self.active_adapter = adapter_name
        self._disable_adapters = False

        if r > 0:
            self.add_adapter(adapter_name, r=r, lora_alpha=lora_alpha)

    def add_adapter(self, adapter_name, r, lora_alpha):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r

        self.lora_A[adapter_name] = nn.Conv2d(
            self.in_channels,
            r,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=False
        )

        self.lora_B[adapter_name] = nn.Conv2d(
            r,
            self.out_channels,
            kernel_size=1,
            bias=False
        )

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[adapter_name].weight)

    def set_adapter(self, adapter_name):
        if adapter_name not in self.lora_A:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name

    def enable_adapters(self, enabled=True):
        self._disable_adapters = not enabled

    def forward(self, x):
        result = self.conv(x)
        if self.active_adapter in self.lora_A and not self._disable_adapters:
            lora_A = self.lora_A[self.active_adapter](x)
            lora_B = self.lora_B[self.active_adapter](lora_A)
            result = result + self.scaling[self.active_adapter] * lora_B
        return result

    def get_base_layer(self):
        return self.conv

    def delete_adapter(self, adapter_name):
        if adapter_name in self.lora_A:
            del self.lora_A[adapter_name]
            del self.lora_B[adapter_name]
            del self.r[adapter_name]
            del self.lora_alpha[adapter_name]
            del self.scaling[adapter_name]
