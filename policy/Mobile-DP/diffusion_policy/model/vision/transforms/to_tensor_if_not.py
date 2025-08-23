#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class ToTensorIfNot(T.ToTensor):
    def __call__(self, pic):
        if not torch.is_tensor(pic):
            return super().__call__(pic)
        return pic

class ToTensorModule(torch.nn.Module):
    """Converts a PIL Image or numpy.ndarray to a FloatTensor.
    Converts image to tensor (H x W x C) in the range [0.0, 1.0].
    """
    def __init__(self):
        super(ToTensorModule, self).__init__()

    def forward(self, x):
        return x / 255.0