# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Thinh DV 3 Sept 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.torch_utils import fuse_conv_and_bn
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL","HGBlock","HGStem","SPP","SPPF","C1","C2","C3","C2f","C2fAttn","ImagePoolingAttn",
    "ContrastiveHead","BNContrastiveHead","C3x","C3TR","C3Ghost","GhostBottleneck","Bottleneck",
    "BottleneckCSP","Proto","RepC3","ResNetLayer","RepNCSPELAN4","ELAN1","ADown","AConv","SPPELAN",
    "CBFuse","CBLinear","RepVGGDW","CIB","C2fCIB","Attention","PSA","SCDown",
    
    "PositionalEmbedding", "MyMSA", "MyMLP", "Alexnet", "MyMHSA", "SAM", "CAM", "CBAM", "SEBlock", 
    "ECA", "MyAdd", "MyClassify", "MyClass", "Block1", "Block21", "Block31", "Block41", "Block51", 
    "Block22", "Block32", "Block42", "Block52", "Skip", "MyMHSA_v2", "MyESC", "h_sigmoid", "h_swish", 
    "CoordAtt", "MyESCC", "MyESC2", "MyESC3", "C_Attention", "H_Attention", "W_Attention", "HWC", "HWC2"
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
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
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSA(nn.Module):
    """
    Position-wise Spatial Attention module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float): Expansion factor for the intermediate channels. Default is 0.5.

    Attributes:
        c (int): Number of intermediate channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for spatial attention.
        ffn (nn.Sequential): Feed-forward network module.
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes convolution layers, attention module, and feed-forward network with channel reduction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        Forward pass of the PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class SCDown(nn.Module):
    """Spatial Channel Downsample (SCDown) module for reducing spatial and channel dimensions."""

    def __init__(self, c1, c2, k, s):
        """
        Spatial Channel Downsample (SCDown) module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for the convolutional layer.
            s (int): Stride for the convolutional layer.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """
        Forward pass of the SCDown module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the SCDown module.
        """
        return self.cv2(self.cv1(x))

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
    def __init__(self, c1, k_size=3):
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

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.relu(y)
        y = self.sigmoid(y)

        return x * y.expand_as(x)

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
        #self.c2f = C2f(c2*2, c2)   # nho bo cai nay di nhe thi params moi giam
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

        # return x * y.expand_as(x) # use for MyHWC, MyHWC_v1, MyHWC_v2
        # If y is of shape (1, C, 1, 1) and x is of shape (N, C, H, W), 
        # then y.expand_as(x) will expand y to (N, C, H, W) by replicating its values along the new dimensions.
        return y    # B x C x 1 x 1
    
class H_Attention(nn.Module):
    """Constructs a Height Attention module.
    Input: Batch size x Channel x Height x Width: 1x96x20x20
    Args:
        c2: Number of channels from the previous layer
        kernel_size - k: Adaptive selection of kernel size
        Output: 1 x 2048 x 20 x 1
    """
    def __init__(self, c1, c2, k=3):
        super(H_Attention, self).__init__()
        self.kernel_size = k
        
        # For Height Attention
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # Output size: [H, 1]
        self.conv = nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=k, padding=(k - 1) // 2, bias=False)  # Convolution to maintain the number of channels
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input-x example: 1 x 1024 x 20 x 20
        B, C, H, W = x.shape
        
        # Adaptive Average Pooling to reduce width to 1
        y = self.avg_pool(x)  # y: [B, C, H, 1] 1 x 1024 x 20 x 1
        y = y.squeeze(-1)     # y: [B, C, H]: 1 x 1024 x 20
        
        # # Prepare for Conv1d by permuting dimensions
        # y = y.permute(0, 2, 1)  # y: [B, H, C]
        
        # # Multi-scale information fusion
        y = self.conv(y)  # [B, C, H'] 1 x 2048 x 20
        y = self.silu(y)
        y = self.sigmoid(y)
        
        # Restore the shape [B, C, H, 1]
        # y = y.permute(0, 2, 1)  # [B, C, H', 1]
        y = y.unsqueeze(-1)     # [B, C, H', 1] 1 x 2048 x 20 x 1
        
        return y  # Output shape: [B, C, H', 1]: 1 x 2048 x 20 x 1

class W_Attention(nn.Module):
    """Constructs a Height Attention module.
    Input: Batch size x Channel x Height x Width: 1x96x20x20
    Args:
        c2: Number of channels from the previous layer
        kernel_size - k: Adaptive selection of kernel size
        Output: 1 x 2048 x 1 x 20
    """
    def __init__(self, c1, c2, k=3):
        super(W_Attention, self).__init__()
        self.kernel_size = k
        
        # For Height Attention
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))  # Output size: [1, W]
        self.conv = nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=k, padding=(k - 1) // 2, bias=False)  # Convolution to maintain the number of channels
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input-x example: 1 x 1024 x 20 x 20
        B, C, H, W = x.shape
        
        # Adaptive Average Pooling to reduce width to 1
        y = self.avg_pool(x)  # y: [B, C, 1, W] 1 x 1024 x 1 x 20
        y = y.squeeze(-2)     # y: [B, C, W]: 1 x 1024 x 20
        
        # # Multi-scale information fusion
        y = self.conv(y)  # [B, C, W'] 1 x 2048 x 20
        y = self.silu(y)
        y = self.sigmoid(y)
        
        # Restore the shape [B, C, 1, W]
        y = y.unsqueeze(-2)     # [B, C, 1, W'] 1 x 2048 x 1 x 20
        
        return y  # Output shape: [B, C, 1, W']: 1 x 2048 x 1 x 20
    
class HWC(nn.Module):

    def __init__(self, c1, c2, d): #c1: input chanel, c2: output chanel, d: input channel for MHSA
        super().__init__()
        self.ha = H_Attention(c1, c1)
        self.wa = W_Attention(c1, c1)
        self.ca = C_Attention(c2)
        self.mhsa = MyMHSA_v2(c1,c2,d) # heads=2
        self.conv = nn.Conv2d(c2*2, c2, 1, 1)   # after concate(spatial, channel) with dim=1 (chanel c2) => c2*4

    def forward(self, x):
        print("# Input Size: ", x.size())
        h_att = self.ha(x)                                  # B x C1 x H x 1
        print("# Height Output Size: ", h_att.size())
        w_att = self.wa(x)                                  # B x C1 x 1 x W
        print("# Weight Output Size: ", w_att.size())
        hw_att = h_att * w_att                              # B x C1 x H x W   
        print("# H*W Output Size: ", hw_att.size())   
        spa_att = self.mhsa(hw_att)                         # B x 2*C1=C2 x H x W
        print("# Spatial Output Size: ", spa_att.size())
        cha_att = self.ca(self.mhsa((x)))                   # B x 2*C1=C2 x H x W
        print("# Channel Output Size: ", cha_att.size())
        
        hwc_out = torch.cat((spa_att, cha_att),dim=1)       # B x 2*C2 x H x W
        print("# HWC Output Size: ", hwc_out.size())  
        hwc_out = self.conv(hwc_out)                        # B x C2 x H x W 
        print("# After Conv2D Output Size: ", hwc_out.size())
        return hwc_out
        #Muc dich su dung cov2d thay c2f la de giam params

class HWC2(nn.Module):

    def __init__(self, c1, c2, d): #c1: input chanel, c2: output chanel, d: input channel for MHSA
        super().__init__()
        self.ha = H_Attention(c1, c1)
        self.wa = W_Attention(c1, c1)
        self.ca = C_Attention(c2)
        self.mhsa = MyMHSA_v2(c1,c2,d) # heads=2
        self.conv = nn.Conv2d(c2*2, c2, 1, 1)   # after concate(spatial, channel) with dim=1 (chanel c2) => c2*4
        self.spp = SPP(c1,c2)

    def forward(self, x):
        print("# Input Size: ", x.size())
        h_att = self.ha(x)                                  # B x C1 x H x 1
        print("# Height Output Size: ", h_att.size())
        w_att = self.wa(x)                                  # B x C1 x 1 x W
        print("# Weight Output Size: ", w_att.size())
        hw_att = h_att * w_att                              # B x C1 x H x W   
        print("# H*W Output Size: ", hw_att.size())   
        spa_att = self.spp(hw_att)                         # B x 2*C1=C2 x H x W
        print("# Spatial Output Size: ", spa_att.size())
        cha_att = self.ca(self.mhsa((x)))                   # B x 2*C1=C2 x H x W
        print("# Channel Output Size: ", cha_att.size())
        
        hwc_out = torch.cat((spa_att, cha_att),dim=1)       # B x 2*C2 x H x W
        print("# HWC Output Size: ", hwc_out.size())  
        hwc_out = self.conv(hwc_out)                        # B x C2 x H x W 
        print("# After Conv2D Output Size: ", hwc_out.size())
        return hwc_out
        #Muc dich su dung cov2d thay c2f la de giam params

class CBS(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, activation)
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
           
class SpaceDownS(nn.Module):    # Input CxHxW
    def __init__(self, c1, c2):
        super().__init__()
        self.ha = H_Attention(c1, c1)
        self.wa = W_Attention(c1, c1)
        self.mp5 = nn.MaxPool2d(kernel_size=5, stride=2, padding=5//2)
        self.mp9 = nn.MaxPool2d(kernel_size=9, stride=2, padding=9//2)
        self.mp13 = nn.MaxPool2d(kernel_size=13, stride=2, padding=13//2)
        self.conv = nn.Conv2d(c1*3, c2, 1, 1)   # after concate 3 maxpoling
        self.bn = nn.BatchNorm2d(c2)
        self.sl = nn.SiLU()
        
    def forward(self, x):
        print("# Input Size: ", x.size())                   # B x C1 x H x W
        h_att = self.ha(x)                                  
        print("# Height Output Size: ", h_att.size())       # B x C1 x H x 1
        w_att = self.wa(x)                                  
        print("# Weight Output Size: ", w_att.size())       # B x C1 x 1 x W
        hw_att = x * h_att * w_att                             
        print("# H*W Output Size: ", hw_att.size())         # B x C1 x H x W
        
        mpl5 = self.mp5(x)
        print("# MP5 Output Size: ", mpl5.size())           # B x C1 x H/2 x W/2
        mpl9 = self.mp9(x)
        print("# MP9 Output Size: ", mpl9.size())           # B x C1 x H/2 x W/2
        mpl13 = self.mp13(x)
        print("# MP13 Output Size: ", mpl13.size())         # B x C1 x H/2 x W/2
        
        cat = torch.cat((mpl5,mpl9,mpl13),dim=1)
        print("# CAT Output Size: ", cat.size())            # B x 3*C1 x H/2 x W/2
        con = self.conv(cat)
        print("# CONV Output Size: ", con.size())           # B x C2 x H/2 x W/2
        spacedown = self.sl(self.bn(con))
        print("# SPD Output Size: ", spacedown.size())      # B x C2 x H/2 x W/2
        return spacedown

class ScaleDotProduct(nn.Module):    # Input CxHxW
    def __init__(self, c1, c2):
        super().__init__()
        self.ha = H_Attention(c1, c1)
        self.wa = W_Attention(c1, c1)
        self.ca = C_Attention(c2)
        
    def forward(self, x):
        # print("# Input Size: ", x.size())                   # B x C1 x H x W
        h_att = self.ha(x)                                  
        # print("# Height Output Size: ", h_att.size())       # B x C1 x H x 1
        w_att = self.wa(x)                                  
        # print("# Weight Output Size: ", w_att.size())       # B x C1 x 1 x W
        c_att = self.ca(x)                                  
        # print("# Channel Output Size: ", c_att.size())      # B x C1 x 1 x 1
        
        # matmul (H,W)
        BQ,CQ,HQ,WQ = h_att.size()
        Q = h_att.view(BQ,CQ,HQ*WQ)
        # print("# Q Output Size: ", Q.size())                # B x C1 x H*W
        
        BK,CK,HK,WK = w_att.size()
        K = w_att.view(BK,CK,HK*WK)
        # print("# K Output Size: ", K.size())                # B x C1 x H*W
        
        # Compute the attention scores by performing matrix multiplication of Q and K
        K_transpose = K.transpose(1,2)
        # print("# K_transpose Output Size: ", K_transpose.size())    # B x H*W x C1
        
        scores = torch.bmm(Q, K_transpose)
        # print("# scores Output Size: ", scores.size())              # B x C1 x C1
        
        # Scale the scores
        d_k = Q.size(-1)  # This is the depth (H*W)
        scores_scaled = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # Scale by sqrt(d_k)
        # print("# Scaled scores Output Size: ", scores_scaled.size())     # B x C1 x C1
        
        # Apply softmax to get the attention weights
        attention_weights = F.softmax(scores_scaled, dim=-1)  # Shape: [Batch, Channels, Channels]
        # print("# Attention weights Output Size: ", attention_weights.size())     # B x C1 x C1

        # Multiply the attention weights with the value vector V
        BV,CV,HV,WV = c_att.size()
        V = c_att.view(BV,CV,HV*WV)
        output = torch.bmm(attention_weights, V)    # Shape: [Batch, Channels, 1]
        # print("# Output Size: ", output.size())     # B x C1 x C1

        # Reshape output to match the original spatial dimensions
        output = output.view(BV, CV, 1, 1)
        # print("# Output Reshape Size: ", output.size())     # B x C1 x 1 x 1
        
        # trans from into input shape
        output = x * output
        # print("# Final Reshape Size: ", output.size())     # B x C1 x H x W
        
        return output   
    
class Contigous_Att(nn.Module):    # Input CxHxW
    def __init__(self, c1, c2):
        super().__init__()
        self.sdp = ScaleDotProduct(c1, c2)
        self.conv = nn.Conv2d(c1*4, c2, 1, 1)   # after concate 3 maxpoling
        self.bn = nn.BatchNorm2d(c2)
        self.sl = nn.SiLU()
        
    def forward(self, x):
        # print("# Input Size: ", x.size())                   # B x C1 x H x W
        y1 = self.sdp(x)                                  
        # print("# y1 Output Size: ", y1.size())              # B x C1 x H x W
        y2 = self.sdp(y1)                                  
        # print("# y2 Output Size: ", y2.size())              # B x C1 x H x W
        y3 = self.sdp(y2)                                  
        # print("# y3 Output Size: ", y3.size())              # B x C1 x H x W
        y4 = self.sdp(y1)                                  
        # print("# y4 Output Size: ", y4.size())              # B x C1 x H x W
        output = torch.cat((y1,y2,y3,y4),dim=1)
        # print("# Output Size: ", output.size())             # B x C1*4 x H x W
        output = self.sl(self.bn(self.conv(output)))
        # print("# Output Final Size: ", output.size())             # B x C1*4 x H x W
        return output   