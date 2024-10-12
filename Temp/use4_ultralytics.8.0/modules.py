# Ultralytics version 8.0.0
"""
env yolov8 (ultra 8.0.0): 27 August 2024
"""
import math
import warnings
from copy import copy
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.cuda import amp
from nn.autobackend import AutoBackend
from yolo.data.augment import LetterBox
from yolo.utils import LOGGER, colorstr
from yolo.utils.files import increment_path
from yolo.utils.ops import Profile, make_divisible, non_max_suppression, scale_boxes, xyxy2xywh
from yolo.utils.plotting import Annotator, colors, save_one_box
from yolo.utils.tal import dist2bbox, make_anchors
from yolo.utils.torch_utils import copy_attr, smart_inference_mode


def autopad(k, p=None, d=1):  # kernel, padding, dilation

    # Pad to 'same' shape outputs

    if d > 1:

        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size

    if p is None:

        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad

    return p

class Conv(nn.Module):

    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)

    default_act = nn.SiLU()  # default activation



    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):

        super().__init__()

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        self.bn = nn.BatchNorm2d(c2)

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()



    def forward(self, x):

        # print("# Conv Output Size: ", self.act(self.bn(self.conv(x))).size())

        return self.act(self.bn(self.conv(x)))



    def forward_fuse(self, x):

        return self.act(self.conv(x))

class DWConv(Conv):

    # Depth-wise convolution

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation

        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class DWConvTranspose2d(nn.ConvTranspose2d):

    # Depth-wise transpose convolution

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out

        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))

class ConvTranspose(nn.Module):

    # Convolution transpose 2d layer

    default_act = nn.SiLU()  # default activation



    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):

        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)

        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()



    def forward(self, x):

        return self.act(self.bn(self.conv_transpose(x)))

class DFL(nn.Module):

    # DFL module

    def __init__(self, c1=16):

        super().__init__()

        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)

        x = torch.arange(c1, dtype=torch.float)

        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))

        self.c1 = c1



    def forward(self, x):

        b, c, a = x.shape  # batch, channels, anchors

        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class TransformerLayer(nn.Module):

    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)

    def __init__(self, c, num_heads):

        super().__init__()

        self.q = nn.Linear(c, c, bias=False)

        self.k = nn.Linear(c, c, bias=False)

        self.v = nn.Linear(c, c, bias=False)

        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)

        self.fc1 = nn.Linear(c, c, bias=False)

        self.fc2 = nn.Linear(c, c, bias=False)



    def forward(self, x):

        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x

        x = self.fc2(self.fc1(x)) + x

        return x

class TransformerBlock(nn.Module):

    # Vision Transformer https://arxiv.org/abs/2010.11929

    def __init__(self, c1, c2, num_heads, num_layers):

        super().__init__()

        self.conv = None

        if c1 != c2:

            self.conv = Conv(c1, c2)

        self.linear = nn.Linear(c2, c2)  # learnable position embedding

        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))

        self.c2 = c2



    def forward(self, x):

        if self.conv is not None:

            x = self.conv(x)

        b, _, w, h = x.shape

        p = x.flatten(2).permute(2, 0, 1)

        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

class Bottleneck(nn.Module):

    # Standard bottleneck

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand

        super().__init__()

        c_ = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, c_, k[0], 1)

        self.cv2 = Conv(c_, c2, k[1], 1, g=g)

        self.add = shortcut and c1 == c2



    def forward(self, x):

        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):

    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion

        super().__init__()

        c_ = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1)

        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)

        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)

        self.cv4 = Conv(2 * c_, c2, 1, 1)

        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)

        self.act = nn.SiLU()

        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))



    def forward(self, x):

        y1 = self.cv3(self.m(self.cv1(x)))

        y2 = self.cv2(x)

        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class C3(nn.Module):

    # CSP Bottleneck with 3 convolutions

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion

        super().__init__()

        c_ = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1)

        self.cv2 = Conv(c1, c_, 1, 1)

        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)

        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))



    def forward(self, x):

        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C2(nn.Module):

    # CSP Bottleneck with 2 convolutions

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion

        super().__init__()

        self.c = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)

        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()

        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))



    def forward(self, x):

        a, b = self.cv1(x).split((self.c, self.c), 1)

        return self.cv2(torch.cat((self.m(a), b), 1))

class C2f(nn.Module):

    # CSP Bottleneck with 2 convolutions

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion

        super().__init__()

        self.c = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)

        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))



    def forward(self, x):

        y = list(self.cv1(x).split((self.c, self.c), 1))

        y.extend(m(y[-1]) for m in self.m)

        # print("# C2F Output Size: ", self.cv2(torch.cat(y, 1)).size())

        return self.cv2(torch.cat(y, 1))

class ChannelAttention(nn.Module):

    # Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet

    def __init__(self, channels: int) -> None:

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

        self.act = nn.Sigmoid()



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x * self.act(self.fc(self.pool(x)))

class SpatialAttention(nn.Module):

    # Spatial-attention module

    def __init__(self, kernel_size=7):

        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'

        padding = 3 if kernel_size == 7 else 1

        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.act = nn.Sigmoid()



    def forward(self, x):

        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class CBAM(nn.Module):

    # CSP Bottleneck with 3 convolutions

    def __init__(self, c1, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion

        super().__init__()

        self.channel_attention = ChannelAttention(c1)

        self.spatial_attention = SpatialAttention(kernel_size)



    def forward(self, x):

        return self.spatial_attention(self.channel_attention(x))

class C1(nn.Module):

    # CSP Bottleneck with 3 convolutions

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number, shortcut, groups, expansion

        super().__init__()

        self.cv1 = Conv(c1, c2, 1, 1)

        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))



    def forward(self, x):

        y = self.cv1(x)

        return self.m(y) + y

class C3x(C3):

    # C3 module with cross-convolutions

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):

        super().__init__(c1, c2, n, shortcut, g, e)

        self.c_ = int(c2 * e)

        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))

class C3TR(C3):

    # C3 module with TransformerBlock()

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):

        super().__init__(c1, c2, n, shortcut, g, e)

        c_ = int(c2 * e)

        self.m = TransformerBlock(c_, c_, 4, n)

class C3Ghost(C3):

    # C3 module with GhostBottleneck()

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):

        super().__init__(c1, c2, n, shortcut, g, e)

        c_ = int(c2 * e)  # hidden channels

        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

class SPP(nn.Module):

    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729

    def __init__(self, c1, c2, k=(5, 9, 13)):

        super().__init__()

        c_ = c1 // 2  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1)

        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)

        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])



    def forward(self, x):

        x = self.cv1(x)

        with warnings.catch_warnings():

            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning

            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPF(nn.Module):

    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))

        super().__init__()

        c_ = c1 // 2  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1)

        self.cv2 = Conv(c_ * 4, c2, 1, 1)

        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)



    def forward(self, x):

        x = self.cv1(x)

        with warnings.catch_warnings():

            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning

            y1 = self.m(x)

            y2 = self.m(y1)

            # print("# SPPF Output Size: ", self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1)).size())

            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Focus(nn.Module):

    # Focus wh information into c-space

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups

        super().__init__()

        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)

        # self.contract = Contract(gain=2)



    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)

        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))

class GhostConv(nn.Module):

    # Ghost Convolution https://github.com/huawei-noah/ghostnet

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups

        super().__init__()

        c_ = c2 // 2  # hidden channels

        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)

        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)



    def forward(self, x):

        y = self.cv1(x)

        return torch.cat((y, self.cv2(y)), 1)

class GhostBottleneck(nn.Module):

    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride

        super().__init__()

        c_ = c2 // 2

        self.conv = nn.Sequential(

            GhostConv(c1, c_, 1, 1),  # pw

            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw

            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear

        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,

                                                                            act=False)) if s == 2 else nn.Identity()



    def forward(self, x):

        return self.conv(x) + self.shortcut(x)

class Concat(nn.Module):

    # Concatenate a list of tensors along dimension

    def __init__(self, dimension=1):

        super().__init__()

        self.d = dimension



    def forward(self, x):

        # print("# Concat Output Size: ", torch.cat(x, self.d).size())

        return torch.cat(x, self.d)

class AutoShape(nn.Module):

    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS

    conf = 0.25  # NMS confidence threshold

    iou = 0.45  # NMS IoU threshold

    agnostic = False  # NMS class-agnostic

    multi_label = False  # NMS multiple labels per box

    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs

    max_det = 1000  # maximum number of detections per image

    amp = False  # Automatic Mixed Precision (AMP) inference



    def __init__(self, model, verbose=True):

        super().__init__()

        if verbose:

            LOGGER.info('Adding AutoShape... ')

        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes

        self.dmb = isinstance(model, AutoBackend)  # DetectMultiBackend() instance

        self.pt = not self.dmb or model.pt  # PyTorch model

        self.model = model.eval()

        if self.pt:

            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()

            m.inplace = False  # Detect.inplace=False for safe multithread inference

            m.export = True  # do not output loss values



    def _apply(self, fn):

        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers

        self = super()._apply(fn)

        if self.pt:

            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()

            m.stride = fn(m.stride)

            m.grid = list(map(fn, m.grid))

            if isinstance(m.anchor_grid, list):

                m.anchor_grid = list(map(fn, m.anchor_grid))

        return self



    @smart_inference_mode()

    def forward(self, ims, size=640, augment=False, profile=False):

        # Inference from various sources. For size(height=640, width=1280), RGB images example inputs are:

        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath

        #   URI:             = 'https://com/images/zidane.jpg'

        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)

        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)

        #   numpy:           = np.zeros((640,1280,3))  # HWC

        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)

        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images



        dt = (Profile(), Profile(), Profile())

        with dt[0]:

            if isinstance(size, int):  # expand

                size = (size, size)

            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param

            autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference

            if isinstance(ims, torch.Tensor):  # torch

                with amp.autocast(autocast):

                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference



            # Pre-process

            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images

            shape0, shape1, files = [], [], []  # image and inference shapes, filenames

            for i, im in enumerate(ims):

                f = f'image{i}'  # filename

                if isinstance(im, (str, Path)):  # filename or uri

                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im

                    im = np.asarray(ImageOps.exif_transpose(im))

                elif isinstance(im, Image.Image):  # PIL Image

                    im, f = np.asarray(ImageOps.exif_transpose(im)), getattr(im, 'filename', f) or f

                files.append(Path(f).with_suffix('.jpg').name)

                if im.shape[0] < 5:  # image in CHW

                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)

                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input

                s = im.shape[:2]  # HWC

                shape0.append(s)  # image shape

                g = max(size) / max(s)  # gain

                shape1.append([y * g for y in s])

                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update

            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)] if self.pt else size  # inf shape

            x = [LetterBox(shape1, auto=False)(image=im)["img"] for im in ims]  # pad

            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW

            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32



        with amp.autocast(autocast):

            # Inference

            with dt[1]:

                y = self.model(x, augment=augment)  # forward



            # Post-process

            with dt[2]:

                y = non_max_suppression(y if self.dmb else y[0],

                                        self.conf,

                                        self.iou,

                                        self.classes,

                                        self.agnostic,

                                        self.multi_label,

                                        max_det=self.max_det)  # NMS

                for i in range(n):

                    scale_boxes(shape1, y[i][:, :4], shape0[i])



            return Detections(ims, y, files, dt, self.names, x.shape)

class Detections:

    # YOLOv5 detections class for inference results

    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):

        super().__init__()

        d = pred[0].device  # device

        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations

        self.ims = ims  # list of images as numpy arrays

        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)

        self.names = names  # class names

        self.files = files  # image filenames

        self.times = times  # profiling times

        self.xyxy = pred  # xyxy pixels

        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels

        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized

        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized

        self.n = len(self.pred)  # number of images (batch size)

        self.t = tuple(x.t / self.n * 1E3 for x in times)  # timestamps (ms)

        self.s = tuple(shape)  # inference BCHW shape



    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):

        s, crops = '', []

        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):

            s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string

            if pred.shape[0]:

                for c in pred[:, -1].unique():

                    n = (pred[:, -1] == c).sum()  # detections per class

                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                s = s.rstrip(', ')

                if show or save or render or crop:

                    annotator = Annotator(im, example=str(self.names))

                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class

                        label = f'{self.names[int(cls)]} {conf:.2f}'

                        if crop:

                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None

                            crops.append({

                                'box': box,

                                'conf': conf,

                                'cls': cls,

                                'label': label,

                                'im': save_one_box(box, im, file=file, save=save)})

                        else:  # all others

                            annotator.box_label(box, label if labels else '', color=colors(cls))

                    im = annotator.im

            else:

                s += '(no detections)'



            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np

            if show:

                im.show(self.files[i])  # show

            if save:

                f = self.files[i]

                im.save(save_dir / f)  # save

                if i == self.n - 1:

                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")

            if render:

                self.ims[i] = np.asarray(im)

        if pprint:

            s = s.lstrip('\n')

            return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t

        if crop:

            if save:

                LOGGER.info(f'Saved results to {save_dir}\n')

            return crops



    def show(self, labels=True):

        self._run(show=True, labels=labels)  # show results



    def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):

        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir

        self._run(save=True, labels=labels, save_dir=save_dir)  # save results



    def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):

        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None

        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results



    def render(self, labels=True):

        self._run(render=True, labels=labels)  # render results

        return self.ims



    def pandas(self):

        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])

        new = copy(self)  # return copy

        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns

        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns

        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):

            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update

            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])

        return new



    def tolist(self):

        # return a list of Detections objects, i.e. 'for result in results.tolist():'

        r = range(self.n)  # iterable

        x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]

        # for d in x:

        #    for k in ['ims', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:

        #        setattr(d, k, getattr(d, k)[0])  # pop out of list

        return x



    def print(self):

        LOGGER.info(self.__str__())



    def __len__(self):  # override len(results)

        return self.n



    def __str__(self):  # override print(results)

        return self._run(pprint=True)  # print results



    def __repr__(self):

        return f'YOLOv5 {self.__class__} instance\n' + self.__str__()

class Proto(nn.Module):

    # YOLOv8 mask Proto module for segmentation models

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks

        super().__init__()

        self.cv1 = Conv(c1, c_, k=3)

        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')

        self.cv2 = Conv(c_, c_, k=3)

        self.cv3 = Conv(c_, c2)



    def forward(self, x):

        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class Ensemble(nn.ModuleList):

    # Ensemble of models

    def __init__(self):

        super().__init__()



    def forward(self, x, augment=False, profile=False, visualize=False):

        y = [module(x, augment, profile, visualize)[0] for module in self]

        # y = torch.stack(y).max(0)[0]  # max ensemble

        # y = torch.stack(y).mean(0)  # mean ensemble

        y = torch.cat(y, 1)  # nms ensemble

        return y, None  # inference, train output

# heads

class Detect(nn.Module):

    # YOLOv5 Detect head for detection models

    dynamic = False  # force grid reconstruction

    export = False  # export mode

    shape = None

    anchors = torch.empty(0)  # init

    strides = torch.empty(0)  # init



    def __init__(self, nc=80, ch=()):  # detection layer

        super().__init__()

        self.nc = nc  # number of classes

        self.nl = len(ch)  # number of detection layers

        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)

        self.no = nc + self.reg_max * 4  # number of outputs per anchor

        self.stride = torch.zeros(self.nl)  # strides computed during build



        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels

        self.cv2 = nn.ModuleList(

            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)

        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()



    def forward(self, x):

        shape = x[0].shape  # BCHW

        for i in range(self.nl):

            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:

            return x

        elif self.dynamic or self.shape != shape:

            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

            self.shape = shape



        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)

        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)

        return y if self.export else (y, x)



    def bias_init(self):

        # Initialize Detect() biases, WARNING: requires stride availability

        m = self  # self.model[-1]  # Detect() module

        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1

        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency

        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from

            a[-1].bias.data[:] = 1.0  # box

            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

class Segment(Detect):

    # YOLOv5 Segment head for segmentation models

    def __init__(self, nc=80, nm=32, npr=256, ch=()):

        super().__init__(nc, ch)

        self.nm = nm  # number of masks

        self.npr = npr  # number of protos

        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        self.detect = Detect.forward



        c4 = max(ch[0] // 4, self.nm)

        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)



    def forward(self, x):

        p = self.proto(x[0])  # mask protos

        bs = p.shape[0]  # batch size



        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients

        x = self.detect(self, x)

        if self.training:

            return x, mc, p

        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))

class Classify(nn.Module):

    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups

        super().__init__()

        c_ = 1280  # efficientnet_b0 size

        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)

        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)

        self.drop = nn.Dropout(p=0.0, inplace=True)

        self.linear = nn.Linear(c_, c2)  # to x(b,c2)



    def forward(self, x):

        if isinstance(x, list):

            x = torch.cat(x, 1)

        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

# ----------------- MODIFY WITH ATTENTION AND SO ON --------- # 
class MyClassify(nn.Module): # MLP duoc thiet ke boi chinh tui
    
    # Edited for comfortable with ViT by me
    def __init__(self, c1=512, c2=512, num_classes = 53):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 6 * 6, num_classes), 
            #  512 = output chanel of concate(Alexnet, MHSA) = ((256+256)x6x6), 6 = output of Alex feature map out
        )

    def forward(self, x):
        x = self.model(x)
        return x

class SEBlock(nn.Module):

    def __init__(self, input_channels, reduction_ratio=16):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels, bias=False)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        batch_size, channels, _, _ = x.size()

        y = self.avg_pool(x).view(batch_size, channels)

        y = self.fc1(y)

        y = self.relu(y)

        y = self.fc2(y)

        y = self.sigmoid(y).view(batch_size, channels, 1, 1)

        return x * y.expand_as(x)

class Block1(nn.Module):  # BLOCK-1 (starting block) asume input=(224x224) output=(56x56) chanel1 = 3, chanel2= 64

    default_act = nn.ReLU()

    dropout_percentage = 0.5



    def __init__(self, c1, c2):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.batchnorm1 = nn.BatchNorm2d(c2)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))



    def forward(self, x):

        x = self.conv1(x)

        x = self.batchnorm1(x)

        x = self.maxpool1(x)

        # print("# Block-1 Output Size: ", x.size())

        return x

class Block21(nn.Module):  # BLOCK-2 (1) input=(56x56) output = (56x56) chanel1 = 64, chanel2= 64

    default_act = nn.ReLU()

    dropout_percentage = 0.5



    def __init__(self, c1, c2):

        super(Block21, self).__init__()



        self.conv2_1_1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm2_1_1 = nn.BatchNorm2d(c2)

        self.act2_1 = self.default_act

        self.conv2_1_2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm2_1_2 = nn.BatchNorm2d(c2)

        self.dropout2_1 = nn.Dropout(self.dropout_percentage)

        self.se2_1 = SEBlock(c2)



    def forward(self, x):

        x = self.conv2_1_1(x)

        x = self.batchnorm2_1_1(x)

        x = self.act2_1(x)

        x = self.conv2_1_2(x)

        x = self.batchnorm2_1_2(x)

        x = self.dropout2_1(x)

        x = self.se2_1(x)

        # print("# Block-21 Output Size: ", x.size())

        return x

class Block22(nn.Module):  # BLOCK-2 (2) input=(56x56) output = (56x56) chanel1 = 64, chanel2= 64

    default_act = nn.ReLU()

    dropout_percentage = 0.5



    def __init__(self, c1, c2):

        super(Block22, self).__init__()



        self.conv2_2_1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm2_2_1 = nn.BatchNorm2d(c2)

        self.act2_2 = self.default_act

        self.conv2_2_2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm2_2_2 = nn.BatchNorm2d(c2)

        self.dropout2_2 = nn.Dropout(self.dropout_percentage)

        self.se2_2 = SEBlock(c2)



    def forward(self, x):

        x = self.conv2_2_1(x)

        x = self.batchnorm2_2_1(x)

        x = self.act2_2(x)

        x = self.conv2_2_2(x)

        x = self.batchnorm2_2_2(x)

        x = self.dropout2_2(x)

        x = self.se2_2(x)

        # print("# Block-22 Output Size: ", x.size())

        return x

class Block31(nn.Module):  # BLOCK-3 (1) input=(56x56) output = (28x28) chanel1 = 64, chanel2= 128

    default_act = nn.ReLU()

    dropout_percentage = 0.5



    def __init__(self, c1, c2):

        super(Block31, self).__init__()



        self.conv3_1_1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.batchnorm3_1_1 = nn.BatchNorm2d(c2)

        self.act3_1 = self.default_act

        self.conv3_1_2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm3_1_2 = nn.BatchNorm2d(c2)

        self.dropout3_1 = nn.Dropout(self.dropout_percentage)

        self.se3_1 = SEBlock(c2)



    def forward(self, x):

        x = self.conv3_1_1(x)

        x = self.batchnorm3_1_1(x)

        x = self.act3_1(x)

        x = self.conv3_1_2(x)

        x = self.batchnorm3_1_2(x)

        x = self.dropout3_1(x)

        x = self.se3_1(x)

        # print("# Block-31 Output Size: ", x.size())

        return x

class Block32(nn.Module):  # BLOCK-3 (2) input=(28x28) output = (28x28) chanel1 = 128, chanel2= 128

    default_act = nn.ReLU()

    dropout_percentage = 0.5



    def __init__(self, c1, c2):

        super(Block32, self).__init__()



        self.conv3_2_1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm3_2_1 = nn.BatchNorm2d(c2)

        self.act3_2 = self.default_act

        self.conv3_2_2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm3_2_2 = nn.BatchNorm2d(c2)

        self.dropout3_2 = nn.Dropout(p=self.dropout_percentage)

        self.se3_2 = SEBlock(c2)



    def forward(self, x):

        x = self.conv3_2_1(x)

        x = self.batchnorm3_2_1(x)

        x = self.act3_2(x)

        x = self.conv3_2_2(x)

        x = self.batchnorm3_2_2(x)

        x = self.dropout3_2(x)

        x = self.se3_2(x)

        # print("# Block-32 Output Size: ", x.size())

        return x

class Block41(nn.Module):  # BLOCK-4 (1) input=(28x28) output = (14x14) chanel1 = 128, chanel2= 256

    default_act = nn.ReLU()

    dropout_percentage = 0.5



    def __init__(self, c1, c2):

        super(Block41, self).__init__()



        self.conv4_1_1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.batchnorm4_1_1 = nn.BatchNorm2d(c2)

        self.act4_1 = self.default_act

        self.conv4_1_2 = nn.Conv2d(in_channels=c2, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm4_1_2 = nn.BatchNorm2d(c2)

        self.dropout4_1 = nn.Dropout(self.dropout_percentage)

        self.se4_1 = SEBlock(c2)



    def forward(self, x):

        x = self.conv4_1_1(x)

        x = self.batchnorm4_1_1(x)

        x = self.act4_1(x)

        x = self.conv4_1_2(x)

        x = self.batchnorm4_1_2(x)

        x = self.dropout4_1(x)

        x = self.se4_1(x)

        # print("# Block-41 Output Size: ", x.size())

        return x

class Block42(nn.Module):  # BLOCK-4 (2) input=(14x14) output = (14x14) chanel1 = 256, chanel2= 256

    default_act = nn.ReLU()

    dropout_percentage = 0.5



    def __init__(self, c1, c2):

        super(Block42, self).__init__()



        self.conv4_2_1 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm4_2_1 = nn.BatchNorm2d(c2)

        self.act4_2 = self.default_act

        self.conv4_2_2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm4_2_2 = nn.BatchNorm2d(c2)

        self.dropout4_2 = nn.Dropout(p=self.dropout_percentage)

        self.se4_2 = SEBlock(c2)



    def forward(self, x):

        x = self.conv4_2_1(x)

        x = self.batchnorm4_2_1(x)

        x = self.act4_2(x)

        x = self.conv4_2_2(x)

        x = self.batchnorm4_2_2(x)

        x = self.dropout4_2(x)

        x = self.se4_2(x)

        # print("# Block-42 Output Size: ", x.size())

        return x

class Block51(nn.Module):  # BLOCK-5 (1) input=(14x14) output = (7x7) chanel1 = 256, chanel2= 512

    default_act = nn.ReLU()

    dropout_percentage = 0.5



    def __init__(self, c1, c2):

        super(Block51, self).__init__()



        self.conv5_1_1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.batchnorm5_1_1 = nn.BatchNorm2d(c2)

        self.act5_1 = self.default_act

        self.conv5_1_2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm5_1_2 = nn.BatchNorm2d(c2)

        self.dropout5_1 = nn.Dropout(self.dropout_percentage)

        self.se5_1 = SEBlock(c2)



    def forward(self, x):

        x = self.conv5_1_1(x)

        x = self.batchnorm5_1_1(x)

        x = self.act5_1(x)

        x = self.conv5_1_2(x)

        x = self.batchnorm5_1_2(x)

        x = self.dropout5_1(x)

        x = self.se5_1(x)

        # print("# Block-51 Output Size: ", x.size())

        return x

class Block52(nn.Module):  # BLOCK-5 (2) input=(7x7) output = (7x7) chanel1 = 512, chanel2= 512

    default_act = nn.ReLU()

    dropout_percentage = 0.5



    def __init__(self, c1, c2):

        super(Block52, self).__init__()



        self.conv5_2_1 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm5_2_1 = nn.BatchNorm2d(c2)

        self.act5_2 = self.default_act

        self.conv5_2_2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm5_2_2 = nn.BatchNorm2d(c2)

        self.dropout5_2 = nn.Dropout(p=self.dropout_percentage)

        self.se5_2 = SEBlock(c2)



    def forward(self, x):

        x = self.conv5_2_1(x)

        x = self.batchnorm5_2_1(x)

        x = self.act5_2(x)

        x = self.conv5_2_2(x)

        x = self.batchnorm5_2_2(x)

        x = self.dropout5_2(x)

        x = self.se5_2(x)

        # print("# Block-52 Output Size: ", x.size())

        return x

class MyAdd(nn.Module):

    default_act = nn.ReLU()



    #  Add two tensors

    def __init__(self, arg):

        super(MyAdd, self).__init__()

        self.arg = arg



    def forward(self, x):

        x = torch.add(x[0], x[1])

        x = self.default_act(x)

        # print("# MyAdd Output Size: ", x.size())

        return x

class Skip(nn.Module):

    default_act = nn.ReLU()



    #  Add two tensors

    def __init__(self, c1, c2):

        super(Skip, self).__init__()



        self.skip1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))



    def forward(self, x):

        x = self.skip1(x)

        # print("# Skip Connection Output Size: ", x.size())

        return x

class MyClass(nn.Module):

    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, activation)

    default_act = nn.ReLU()  # default activation



    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):

        super().__init__()

        self.conv = nn.Conv2d(c1, c2, k, s)

        self.bn = nn.BatchNorm2d(c2)

        self.act = self.default_act



    def forward(self, x):

        print("# MyConv Output Size: ", self.act(self.bn(self.conv(x))).size())

        return self.act(self.bn(self.conv(x)))

# ----------- CUC -------------- #
class SEBlock(nn.Module):
    def __init__(self, c1, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(c1, c1 // reduction_ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(c1 // reduction_ratio, c1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)

class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, c2, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        
# Positional Embedding Function
class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, d, device='cpu'):
        # d: output of last C2f = 1024
        # sequence_length: output width, height of feature map = 7x7x1024
        self.sequence_length = sequence_length
        self.d = d
        self.device = device

    def get_embeddings(self):
        result = torch.ones(self.sequence_length, self.d, device=self.device)
        for i in range(self.sequence_length):
            for j in range(self.d):
                result[i][j] = (
                    np.sin(i / (10000 ** (j / self.d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / self.d)))
                )
        return result

# Multi-Head Self Attention (MHSA)
class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):  # d: output channel of position embedding (256)
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        sequences = sequences.to(self.q_mappings[0].weight.dtype)  # Ensure the input is the same dtype as the linear layers

        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]
                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.stack(result)

# MLP Classifier
class MyMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 6 * 6, num_classes), 
            #  512 = output chanel of concate(Alexnet, MHSA) = ((256+256)x6x6), 6 = output of Alex feature map out
        )

    def forward(self, x):
        x = self.model(x)
        return x

# Alexnet
class Alexnet(nn.Module):
    def __init__(self, in_chanel, out_chanel): #3: input chanel, 256: output chanel
        super().__init__()
        self.model = nn.Sequential(
            #### Convolutional Layers ####
            #input: 224*224*3
            # Layer 1:
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4)),  # Change 1 to 3 for RGB images: output = 32
            #output: W = (224-11+2*0)/4 + 1=54, H = (224-11+2*0)/4 + 1=54
            nn.ReLU(),  # output:96, W=H=54 do co cung chieu dai moi canh sau conv
            nn.BatchNorm2d(96),
            nn.MaxPool2d(2, 2),
            # after pooling: ((W=H=54-kernel=2)/stride=2) + 1= 27; Neu nn.Flatten(), nn.Linear(96*(27)*(27), 53),

            # Layer 2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=(2, 2)),  # input: 96, output: 256
            #output: W=(27 - 5 + 2 * 2) / 1 + 1 = 26, H=(27 - 5 + 2 * 2) / 1 + 1 = 26
            nn.ReLU(),  # output:256, W=H=26 do co cung chieu dai moi canh sau conv
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 2),
            # after pooling: ((W=H=26-kernel=3)/stride=2)+1 = 12; Neu nn.Flatten(), nn.Linear(256*(12)*(12), 53),

            # Layer 3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1)),  # input: 256, output: 384
            # output: W=(12 - 3 + 2 * 1) / 1 + 1 = 12, W=(12 - 3 + 2 * 1) / 1 + 1 = 12
            nn.ReLU(),  # output:384, W=H=12 do co cung chieu dai moi canh sau conv
            nn.BatchNorm2d(384),

            # Layer 4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=(1, 1)),  # input: 384, output: 384
            # output: W=(12 - 3 + 2 * 1) / 1 + 1 = 12, W=(12 - 3 + 2 * 1) / 1 + 1 = 12
            nn.ReLU(),  # output:384, W=H=12 do co cung chieu dai moi canh sau conv
            nn.BatchNorm2d(384),

            # Layer 5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),  # input: 384, output: 256
            # output: W=(12 - 3 + 2 * 1) / 1 + 1 = 12, W=(12 - 3 + 2 * 1) / 1 + 1 = 12
            nn.ReLU(),  # output:256, W=H=12 do co cung chieu dai moi canh sau conv
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, 2),
            # after pooling: ((W=H=12-kernel=3)/stride=2)+1 = 6; Neu nn.Flatten(), nn.Linear(256*(6)*(6), 53),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class MyMHSA(nn.Module):
    # input from last C2f: 7x7x1024
    # c1: get from C2f
    def __init__(self, c1, c2, sequence_length, d): 
        # c1: chanel image input  (3), c2: MSHA output chanel (512) sau khi concate va se
        # num_classes: number of labels
        # sequence_length: width (height) output C2f (7x7x1024) 
        # d: output chanel from C2f = c1 (1024)
        super().__init__()
        # input from C2f cuoi cung: 7x7x1024
        #self.positional_embedding = PositionalEmbedding(sequence_length, d)
        self.MHSA = MyMSA(d, 2) # heads=2
        
        # After concat (MHSA, C2f): output chanel = d=1024 + 1024. Trong truong hop nay la c2
        # self.SE = SEBlock(c1 + d)
        self.ECA = ECA(c1 + d)

    def forward(self, x):
        # input from last C2f
        
        # Flatten the convolutional output and add positional embeddings
        batch_size, channels, height, width = x.size()
        flat_x = x.view(batch_size, channels, height * width).permute(0, 2, 1)

        # Create positional embeddings on the same device as the input tensor
        #pos_embed = self.positional_embedding.get_embeddings().to(x.device).unsqueeze(0).expand(batch_size, -1, -1)
        #flat_x = flat_x + pos_embed

        # Multi Head Self Attention
        mhsa_output = self.MHSA(flat_x)
        
        # Reshape MHSA output to match the original feature map dimensions
        mhsa_output = mhsa_output.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        # Concatenate the original feature map and MHSA output along the channel dimension
        concat_x = torch.cat((x, mhsa_output), dim=1)  # Concatenate along the channel dimension
        
        ## Adding SE Block
        # seblock = self.SE(concat_x)
        # return seblock
        
        # Adding ECA Block
        eca = self.ECA(concat_x)

        # Pass through the classifier, YOLO ganh team
        #print("# ECA Output Size: ", eca.size())
        return eca

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1,
                              bias=self.bias)

    def forward(self, x):
        max = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        concat = torch.cat((max, avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x
        #print("# SAM Output Size: ", output.size())
        return output

class CAM(nn.Module):
    def __init__(self, channels, r=16):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels // self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels // self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b, c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b, c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        #print("# CAM Output Size: ", output.size())
        return output

class MyMHSA_v2(nn.Module):
    # input from last C2f: 7x7x1024
    # c1: get from C2f
    def __init__(self, c1, c2, d): 
        # c1: chanel image input  (3), c2: MSHA output chanel (512)
        # d: output chanel from C2f = c1 (1024)
        super().__init__()
        self.MHSA = MyMSA(d, 2) # heads=2

    def forward(self, x):
        # input from conv1
        
        # Flatten the convolutional output and add positional embeddings
        batch_size, channels, height, width = x.size()
        flat_x = x.view(batch_size, channels, height * width).permute(0, 2, 1)

        # Multi Head Self Attention
        mhsa_output = self.MHSA(flat_x)
        
        # Reshape MHSA output to match the original feature map dimensions
        mhsa_output = mhsa_output.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        # Concatenate the original feature map and MHSA output along the channel dimension
        concat_x = torch.cat((x, mhsa_output), dim=1)  # Concatenate along the channel dimension
        #print("# MHSA Output Size: ", mhsa_output.size()) #x = 1024, mhsa = 1024 => out = 2048
        return concat_x
              
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, c1, c2, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, c1 // reduction)

        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # origin: 0:batch, 1:chanel, 2:height, 3:width

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class TCA(nn.Module):
    """Constructs a TCA module.
    Args:
        c1: Number of channels of the input feature map: 2048 from MyMHSA_v2
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, c2, k_size=3):
        super(TCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # [batch_size, channels, length]
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # squeeze(-1): remove last dimension
        # transpose(-1, -2): swap 2 last dimensions
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.relu(y)
        y = self.sigmoid(y)

        return x * y.expand_as(x) 
        # If y is of shape (1, C, 1, 1) and x is of shape (N, C, H, W), 
        # then y.expand_as(x) will expand y to (N, C, H, W) by replicating its values along the new dimensions.

class MyESC(nn.Module):

    def __init__(self, c1, c2, d): #c1: input chanel, c2: output chanel
        super().__init__()
        self.sam = SAM()
        self.spp = SPP(c1,c2)
        self.mhsa = MyMHSA_v2(c1,c2,d) # heads=2
        self.c2f = C2f(c2*2, c2)

    def forward(self, x):
        spa_attention = self.spp(self.sam(x))
        #print("# SPA Output Size: ", spa_attention.size())
        cha_attention = self.mhsa((x))
        #print("# CHA Output Size: ", cha_attention.size())
        esc_out = torch.cat((spa_attention,cha_attention),dim=1)
        esc_out = self.c2f(esc_out)
        #print("# MyESC Output Size: ", esc_out.size())
        return esc_out
   
class MyESCC(nn.Module):

    def __init__(self, c1, c2, d): #c1: input chanel, c2: output chanel
        super().__init__()
        self.sam = SAM()
        self.spp = SPP(c1,c2)
        self.mhsa = MyMHSA_v2(c1,c2,d) # heads=2
        self.ca = CoordAtt(c1,c1)
        self.conv = nn.Conv2d(c2*3, c2, 1, 1)   # after cat(spp, mhsa, coordatt) with dim=1 (chanel) => chanel*3

    def forward(self, x):
        spa_attention = self.spp(self.sam(x))                   # 20x20x2048
        #print("# SPA Output Size: ", spa_attention.size())
        cha_attention = self.mhsa(x)                            # 20x20x2048
        #print("# CHA Output Size: ", cha_attention.size())
        coo_attention = torch.cat((x, self.ca(x)), dim=1)       # 20x20x2048
        #print("# CA Output Size: ", coo_attention.size())
        escc_out = torch.cat((spa_attention, cha_attention, coo_attention),dim=1)   # 20x20x2048*3
        escc_out = self.conv(escc_out)                          # 20x20x2048
        #print("# MyESCC Output Size: ", escc_out.size())
        return escc_out

class MyESC3(nn.Module):

    def __init__(self, c1, c2, d): #c1: input chanel, c2: output chanel
        super().__init__()
        self.sam = SAM()
        self.spp = SPP(c1,c2)
        self.mhsa = MyMHSA_v2(c1,c2,d) # heads=2
        self.tca = TCA(c2)
        #self.c2f = C2f(c2*2, c2)   # after cat(spp, mhsa) with dim=1 (chanel) => chanel*2
        self.conv = nn.Conv2d(c2*2, c2, 1, 1)   # after cat(spp, mhsa) with dim=1 (chanel) => chanel*2

    def forward(self, x):
        spa_attention = self.spp(self.sam(x))               
        #print("# SPA Output Size: ", spa_attention.size())              # SPA Output Size:  torch.Size([1, 2048, 8, 8])
        cha_attention = self.tca(self.mhsa((x)))                        # CHA Output Size:  torch.Size([1, 1, 2048, 8, 8])
        #cha_attention = cha_attention.squeeze(1)                        # Remove the second dimension
        #print("# CHA Output Size: ", cha_attention.size())
        esc_out = torch.cat((spa_attention,cha_attention),dim=1)        
        #esc_out = self.c2f(esc_out)         
        esc_out = self.conv(esc_out) 
        #print("# MyESC Output Size: ", esc_out.size())                  # MyESC Output Size:  torch.Size([1, 2048, 8, 8])
        return esc_out
        #Muc dich su dung cov2d thay c2f la de giam params
        
class MyESC2(nn.Module):

    def __init__(self, c1, c2, d): #c1: input chanel, c2: output chanel
        super().__init__()
        self.sam = SAM()
        self.spp = SPP(c1,c2)
        self.mhsa = MyMHSA_v2(c1,c2,d) # heads=2
        #self.c2f = C2f(c2*2, c2)       # nho bo cai nay di thi params moi giam
        self.conv = nn.Conv2d(c2*2, c2, 1, 1)   # after cat(spp, mhsa) with dim=1 (chanel) => chanel*2
        

    def forward(self, x):
        spa_attention = self.spp(self.sam(x))
        #print("# SPA Output Size: ", spa_attention.size())
        cha_attention = self.mhsa((x))
        #print("# CHA Output Size: ", cha_attention.size())
        esc_out = torch.cat((spa_attention,cha_attention),dim=1)
        esc_out = self.conv(esc_out)
        #print("# MyESC Output Size: ", esc_out.size())
        return esc_out

class C_Attention(nn.Module):
    """Constructs a Channel, Height, and Weight Attention module.
    Input: Batch size x Channel x Height x Weight: 1x96x20x20
    Args:
        kernel size - k: Adaptive selection of kernel size
        output: 1x96x1x1
    """
    def __init__(self, c2, k=3): # mac du khong dung c2 nhung ghi cho thong nhat ECA, G_A, W_A
        super(C_Attention, self).__init__()
        # For Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False) # [batch_size, channels, length]
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input-x example: 1x96x20x20
        y = self.avg_pool(x)    # y: 1x96x1x1

        # squeeze(-1): remove last dimension: 1x96x1
        # transpose(-1, -2): swap 2 last dimensions: 1x1x96
        # conv1d(1,1,3,1): kernel size=3 > size(h,w) cua 1x96x1, tuc h=w=1
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.silu(y)
        y = self.sigmoid(y)

        return x * y.expand_as(x) 
        # If y is of shape (1, C, 1, 1) and x is of shape (N, C, H, W), 
        # then y.expand_as(x) will expand y to (N, C, H, W) by replicating its values along the new dimensions.
        #return y    # B x C x 1 x 1
    
class H_Attention(nn.Module):
    """Constructs a Height Attention module.
    Input: Batch size x Channel x Height x Width: 1x96x20x20
    Args:
        c2: Number of channels from the previous layer
        kernel_size - k: Adaptive selection of kernel size
        Output: 1x96x20x1
    """
    def __init__(self, c2, k=3):
        super(H_Attention, self).__init__()
        self.kernel_size = k
        
        # For Height Attention
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # Output size: [H, 1]
        self.conv = nn.Conv1d(in_channels=c2, out_channels=c2, kernel_size=k, padding=(k - 1) // 2, bias=False)  # Convolution to maintain the number of channels
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input-x example: 1x96x20x20
        B, C, H, W = x.shape
        
        # Adaptive Average Pooling to reduce width to 1
        y = self.avg_pool(x)  # y: [B, C, H, 1]
        y = y.squeeze(-1)     # y: [B, C, H]
        
        # Prepare for Conv1d by permuting dimensions
        y = y.permute(0, 2, 1)  # y: [B, H, C]
        
        # Multi-scale information fusion
        y = self.conv(y)  # [B, C, H'] where H' is determined by the convolution
        y = self.silu(y)
        y = self.sigmoid(y)
        
        # Restore the shape [B, C, H, 1]
        y = y.permute(0, 2, 1)  # [B, C, H', 1]
        y = y.unsqueeze(-1)     # [B, C, H', 1]
        
        return y  # Output shape: [B, C, H', 1]: 1x96x20x1

class W_Attention(nn.Module):
    """Constructs a Width Attention module.
    Input: Batch size x Channel x Height x Width: 1x96x20x20
    Args:
        c2: Number of channels from the previous layer
        kernel_size - k: Adaptive selection of kernel size
        Output: 1x96x1x20
    """
    def __init__(self, c2, k=3):
        super(W_Attention, self).__init__()
        self.kernel_size = k
        
        # For Width Attention
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))  # Output size: [1, W]
        self.conv = nn.Conv1d(in_channels=c2, out_channels=c2, kernel_size=k, padding=(k - 1) // 2, bias=False)  # Convolution to maintain the number of channels
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input-x example: 1x96x20x20
        B, C, H, W = x.shape
        
        # Adaptive Average Pooling to reduce height to 1
        y = self.avg_pool(x)  # y: [B, C, 1, W]
        y = y.squeeze(-2)     # y: [B, C, W]
        
        # Prepare for Conv1d by permuting dimensions
        y = y.permute(0, 2, 1)  # y: [B, W, C]
        
        # Multi-scale information fusion
        y = self.conv(y)  # [B, C, W'] where W' is determined by the convolution
        y = self.silu(y)
        y = self.sigmoid(y)
        
        # Restore the shape [B, C, 1, W']
        y = y.permute(0, 2, 1)  # [B, C, W']
        y = y.unsqueeze(-2)     # [B, C, 1, W']
        
        return y  # Output shape: [B, C, 1, W']: 1x96x1x20

class HWC(nn.Module):

    def __init__(self, c1, c2, d): #c1: input chanel, c2: output chanel, d: head for MHSA
        super().__init__()
        self.ha = H_Attention(c2)
        self.wa = W_Attention(c2)
        self.ca = C_Attention(c2)
        self.mhsa = MyMHSA_v2(c1,c2,d) # heads=2
        self.conv = nn.Conv2d(c2*4, c2, 1, 1)   # after concate(spatial, channel) with dim=1 (chanel c2) => c2*4

    def forward(self, x):
        h_att = self.ha(x)                                  # B x C2 x H x 1
        w_att = self.wa(x)                                  # B x C2 x 1 x W
        hw_att = h_att * w_att                              # B x C2 x H x W      
        spa_att = self.mhsa(hw_att)                         # B x 2*C2 x H x W
        
        cha_att = self.ca(self.mhsa((x)))                   # B x 2*C2 x H x W
        
        hwc_out = torch.cat((spa_att, cha_att),dim=1)       # B x 4*C2 x H x W   
        hwc_out = self.conv(hwc_out)                        # B x 2*C2 x H x W 
        return hwc_out
        #Muc dich su dung cov2d thay c2f la de giam params
        
        
        