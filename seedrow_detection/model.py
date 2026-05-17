# =============================================================================
# Corn Seedling Row Detection -- BiSeNet-V2 Model Definition
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Basic Modules
# =============================================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


# =============================================================================
# Detail Branch (1/8 resolution, 128 channels)
# =============================================================================

class DetailBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(3,   64, k=3, s=2, p=1),
            ConvBNReLU(64,  64, k=3, s=1, p=1),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(64,  64, k=3, s=2, p=1),
            ConvBNReLU(64,  64, k=3, s=1, p=1),
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(64,  128, k=3, s=2, p=1),
            ConvBNReLU(128, 128, k=3, s=1, p=1),
        )

    def forward(self, x):
        return self.stage3(self.stage2(self.stage1(x)))


# =============================================================================
# Semantic Branch Components
# =============================================================================

class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv  = ConvBNReLU(3, 16, k=3, s=2, p=1)
        self.left  = nn.Sequential(
            ConvBNReLU(16, 8,  k=1, s=1, p=0),
            ConvBNReLU(8,  16, k=3, s=2, p=1),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse  = ConvBNReLU(32, 16, k=3, s=1, p=1)

    def forward(self, x):
        x = self.conv(x)
        return self.fuse(torch.cat([self.left(x), self.right(x)], dim=1))


class GELayerS1(nn.Module):
    def __init__(self, in_c, out_c, exp=6):
        super().__init__()
        mid = in_c * exp
        self.dw1      = ConvBNReLU(in_c, in_c,  k=3, s=1, p=1, groups=in_c)
        self.conv1    = ConvBNReLU(in_c, mid,   k=1, s=1, p=0)
        self.dw2      = ConvBN    (mid,  mid,   k=3, s=1, p=1, groups=mid)
        self.conv2    = ConvBN    (mid,  out_c, k=1, s=1, p=0)
        self.relu     = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if in_c == out_c else ConvBN(in_c, out_c, k=1, s=1, p=0)

    def forward(self, x):
        skip = self.shortcut(x)
        x    = self.conv2(self.dw2(self.conv1(self.dw1(x))))
        return self.relu(x + skip)


class GELayerS2(nn.Module):
    def __init__(self, in_c, out_c, exp=6):
        super().__init__()
        mid = in_c * exp
        self.dw1      = ConvBNReLU(in_c, in_c,  k=3, s=1, p=1, groups=in_c)
        self.conv1    = ConvBNReLU(in_c, mid,   k=1, s=1, p=0)
        self.dw2      = ConvBN    (mid,  mid,   k=3, s=2, p=1, groups=mid)
        self.conv2    = ConvBN    (mid,  out_c, k=1, s=1, p=0)
        self.relu     = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            ConvBN(in_c, in_c,  k=3, s=2, p=1, groups=in_c),
            ConvBN(in_c, out_c, k=1, s=1, p=0),
        )

    def forward(self, x):
        skip = self.shortcut(x)
        x    = self.conv2(self.dw2(self.conv1(self.dw1(x))))
        return self.relu(x + skip)


class CEBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.bn   = nn.BatchNorm2d(in_c)
        self.conv = ConvBNReLU(in_c, in_c, k=3, s=1, p=1)

    def forward(self, x):
        return self.conv(x + self.bn(self.gap(x)))


class SemanticBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = StemBlock()
        self.stage3 = nn.Sequential(GELayerS2(16, 32), GELayerS1(32, 32))
        self.stage4 = nn.Sequential(GELayerS2(32, 64), GELayerS1(64, 64))
        self.stage5 = nn.Sequential(
            GELayerS2(64,  128), GELayerS1(128, 128),
            GELayerS1(128, 128), GELayerS1(128, 128),
            CEBlock(128),
        )

    def forward(self, x):
        x  = self.stem(x)
        s3 = self.stage3(x)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return s3, s4, s5


# =============================================================================
# BGA Layer (Bilateral Guided Aggregation)
# =============================================================================

class BGALayer(nn.Module):
    def __init__(self, detail_c=128, semantic_c=128, out_c=128):
        super().__init__()
        self.detail_dw   = ConvBN(detail_c,   detail_c, k=3, s=1, p=1, groups=detail_c)
        self.detail_conv = ConvBN(detail_c,   out_c,    k=1, s=1, p=0)
        self.detail_pool = nn.Sequential(
            ConvBN(detail_c, out_c, k=3, s=2, p=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.semantic_conv = ConvBN(semantic_c, out_c, k=1, s=1, p=0)
        self.semantic_dw   = ConvBN(semantic_c, out_c, k=3, s=1, p=1,
                                    groups=semantic_c if semantic_c == out_c else 1)
        self.fuse = ConvBNReLU(out_c, out_c, k=3, s=1, p=1)

    def forward(self, detail, semantic):
        dw  = self.detail_dw(detail)
        dc  = self.detail_conv(dw)
        dp  = self.detail_pool(detail)
        sc  = self.semantic_conv(semantic)
        sdw = self.semantic_dw(semantic)

        sem_up = F.interpolate(torch.sigmoid(dp) * sc,
                               size=detail.shape[2:], mode='bilinear', align_corners=False)
        det_guided = torch.sigmoid(dc) * F.interpolate(
            sdw, size=detail.shape[2:], mode='bilinear', align_corners=False)
        return self.fuse(sem_up + det_guided)


# =============================================================================
# Segmentation Heads
# =============================================================================

class SegHead(nn.Module):
    def __init__(self, in_c, num_classes, scale_factor=8):
        super().__init__()
        self.conv    = ConvBNReLU(in_c, in_c, k=3, s=1, p=1)
        self.dropout = nn.Dropout2d(0.1)
        self.cls     = nn.Conv2d(in_c, num_classes, kernel_size=1)
        self.scale   = scale_factor

    def forward(self, x):
        return F.interpolate(self.cls(self.dropout(self.conv(x))),
                             scale_factor=self.scale, mode='bilinear', align_corners=False)


class AuxHead(nn.Module):
    def __init__(self, in_c, num_classes, scale_factor):
        super().__init__()
        self.conv  = ConvBNReLU(in_c, 64, k=3, s=1, p=1)
        self.cls   = nn.Conv2d(64, num_classes, kernel_size=1)
        self.scale = scale_factor

    def forward(self, x):
        return F.interpolate(self.cls(self.conv(x)),
                             scale_factor=self.scale, mode='bilinear', align_corners=False)


# =============================================================================
# BiSeNet-V2
# =============================================================================

class BiSeNetV2(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.detail   = DetailBranch()
        self.semantic = SemanticBranch()
        self.bga      = BGALayer(detail_c=128, semantic_c=128, out_c=128)
        self.head     = SegHead(128, num_classes, scale_factor=8)
        self.aux3     = AuxHead(32,  num_classes, scale_factor=8)
        self.aux4     = AuxHead(64,  num_classes, scale_factor=16)
        self.aux5     = AuxHead(128, num_classes, scale_factor=32)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        det          = self.detail(x)
        s3, s4, s5  = self.semantic(x)
        fused        = self.bga(det, s5)
        out          = self.head(fused)
        if self.training:
            return out, self.aux3(s3), self.aux4(s4), self.aux5(s5)
        return out


if __name__ == "__main__":
    model = BiSeNetV2(num_classes=2)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {total:.1f} M")

    model.train()
    x    = torch.randn(2, 3, 256, 512)
    outs = model(x)
    print("Train output shapes:", [o.shape for o in outs])

    model.eval()
    with torch.no_grad():
        out = model(x)
    print("Eval  output shape :", out.shape)
