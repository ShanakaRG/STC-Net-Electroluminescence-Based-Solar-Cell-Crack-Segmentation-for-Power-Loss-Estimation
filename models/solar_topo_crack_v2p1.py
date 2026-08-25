from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            else:
                padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden=max(channels // reduction, 4)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Conv2d(channels, hidden, 1)
        self.act=nn.GELU()
        self.fc2=nn.Conv2d(hidden, channels, 1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        w=self.pool(x)
        w=self.fc2(self.act(self.fc1(w)))
        return x * self.sigmoid(w)


class ResidualDWBlock(nn.Module):
    def __init__(self, channels, dilation=1, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(channels * 2, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.bn2(x)
        x = self.se(x)
        x = identity + self.drop_path(x)
        return self.act(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=padding, dilation=dilation,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class FixedSobel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("gx", gx.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.register_buffer("gy", gy.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels

    def forward(self, x):
        gx = F.conv2d(x, self.gx, padding=1, groups=self.in_channels)
        gy = F.conv2d(x, self.gy, padding=1, groups=self.in_channels)
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        return mag


class EdgePyramid(nn.Module):
    def __init__(self, in_channels, feat_channels):
        super().__init__()
        self.sobel = FixedSobel(in_channels)
        self.blocks = nn.ModuleList()
        prev = in_channels
        for out_ch in feat_channels:
            self.blocks.append(
                nn.Sequential(
                    ConvBNAct(prev, out_ch, 3, stride=2),
                    ConvBNAct(out_ch, out_ch, 3, stride=1),
                )
            )
            prev = out_ch

    def forward(self, x):
        edge = self.sobel(x)
        feats = []
        cur = edge
        for block in self.blocks:
            cur = block(cur)
            feats.append(cur)
        return feats


class SpectralResidualMap(nn.Module):
    def __init__(self, cutoff=0.16):
        super().__init__()
        self.cutoff = float(cutoff)
        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer("lap_kernel", lap.view(1, 1, 3, 3))

    def _radial_highpass_mask(self, h, w, device, dtype):
        fy = torch.arange(0, h, device=device, dtype=dtype)
        fx = torch.arange(0, w, device=device, dtype=dtype)
        fy = torch.where(fy <= h // 2, fy, fy - h) / max(float(h), 1.0)
        fx = torch.where(fx <= w // 2, fx, fx - w) / max(float(w), 1.0)
        fy = fy.view(h, 1)
        fx = fx.view(1, w)
        radius = torch.sqrt(fx * fx + fy * fy)
        radius = radius / (radius.max() + 1e-6)
        mask = torch.clamp((radius - self.cutoff) / max(1.0 - self.cutoff, 1e-6), min=0.0, max=1.0)
        return mask.view(1, 1, h, w)

    def forward(self, x):
        x_gray = x.mean(dim=1, keepdim=True)
        if hasattr(torch, "fft") and hasattr(torch.fft, "fft2"):
            h, w = x_gray.shape[-2:]
            mask = self._radial_highpass_mask(h, w, x_gray.device, x_gray.dtype)
            freq = torch.fft.fft2(x_gray)
            filtered = freq * mask
            rec = torch.fft.ifft2(filtered)
            mag = torch.abs(rec)
        else:
            blur = F.avg_pool2d(x_gray, kernel_size=5, stride=1, padding=2)
            mag = torch.abs(x_gray - blur)
            lap = F.conv2d(x_gray, self.lap_kernel.to(x_gray.device, x_gray.dtype), padding=1)
            mag = mag + torch.abs(lap)

        b = mag.shape[0]
        mag = mag.view(b, 1, -1)
        mag = mag - mag.min(dim=2, keepdim=True)[0]
        mag = mag / (mag.max(dim=2, keepdim=True)[0] + 1e-6)
        mag = mag.view(b, 1, x_gray.shape[-2], x_gray.shape[-1])
        return mag


class SpectralPyramid(nn.Module):
    def __init__(self, feat_channels, cutoff=0.16):
        super().__init__()
        self.spectral = SpectralResidualMap(cutoff=cutoff)
        self.blocks = nn.ModuleList()
        prev = 1
        for out_ch in feat_channels:
            self.blocks.append(
                nn.Sequential(
                    ConvBNAct(prev, out_ch, 3, stride=2),
                    ConvBNAct(out_ch, out_ch, 3, stride=1),
                )
            )
            prev = out_ch

    def forward(self, x):
        cur = self.spectral(x)
        feats = []
        for block in self.blocks:
            cur = block(cur)
            feats.append(cur)
        return feats


class MultiScaleContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        dilations = [1, 2, 4, 6]
        self.branches = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv(channels, channels // 2, dilation=d),
                ConvBNAct(channels // 2, channels // 2, 3),
            ) for d in dilations
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, 1, bias=True),
            nn.GELU(),
        )
        fused_in = (len(dilations) + 1) * (channels // 2)
        self.fuse = nn.Sequential(
            ConvBNAct(fused_in, channels, 1),
            ResidualDWBlock(channels, dilation=1),
        )

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(gp)
        return self.fuse(torch.cat(feats, dim=1))


class OrientationAwareMixer(nn.Module):
    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        self.h = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 9), padding=(0, 4), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(9, 1), padding=(4, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            ConvBNAct(channels * 4, channels, 1),
            ResidualDWBlock(channels, dilation=1, drop_path=drop_path),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        merged = torch.cat([self.h(x), self.v(x), self.local(x), self.d1(x)], dim=1)
        fused = self.mix(merged)
        return fused * self.attn(fused) + x


class TopologyAttention(nn.Module):
    def __init__(self, feat_ch, edge_ch, spec_ch):
        super().__init__()
        self.edge_proj = ConvBNAct(edge_ch, feat_ch, 1)
        self.spec_proj = ConvBNAct(spec_ch, feat_ch, 1)
        self.weight_gen = nn.Sequential(
            ConvBNAct(feat_ch * 3, feat_ch, 1),
            nn.Conv2d(feat_ch, feat_ch * 3, 1),
        )
        self.out = nn.Sequential(
            ConvBNAct(feat_ch * 2, feat_ch, 3),
            ResidualDWBlock(feat_ch, dilation=1),
        )

    def forward(self, feat, edge_feat, spec_feat):
        edge_feat = F.interpolate(edge_feat, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        spec_feat = F.interpolate(spec_feat, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        edge_feat = self.edge_proj(edge_feat)
        spec_feat = self.spec_proj(spec_feat)
        merged = torch.cat([feat, edge_feat, spec_feat], dim=1)
        weights = self.weight_gen(merged)
        b, _, h, w = weights.shape
        c = feat.shape[1]
        weights = weights.view(b, 3, c, h, w)
        weights = torch.softmax(weights, dim=1)
        fused = feat * weights[:, 0] + edge_feat * weights[:, 1] + spec_feat * weights[:, 2]
        return self.out(torch.cat([fused, feat], dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, edge_ch, spec_ch):
        super().__init__()
        self.up_proj = ConvBNAct(in_ch, out_ch, 1)
        self.skip_proj = ConvBNAct(skip_ch, out_ch, 1)
        self.topo_attn = TopologyAttention(out_ch, edge_ch=edge_ch, spec_ch=spec_ch)
        self.refine = nn.Sequential(
            ConvBNAct(out_ch * 2, out_ch, 3),
            ResidualDWBlock(out_ch, dilation=1),
            ResidualDWBlock(out_ch, dilation=2),
        )

    def forward(self, x, skip, edge_feat, spec_feat):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_proj(x)
        skip = self.skip_proj(skip)
        skip = self.topo_attn(skip, edge_feat, spec_feat)
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)


class BoundaryTopologyRefiner(nn.Module):
    def __init__(self, feat_ch):
        super().__init__()
        self.mix = nn.Sequential(
            ConvBNAct(feat_ch + 4, feat_ch, 3),
            ResidualDWBlock(feat_ch, dilation=1),
            ConvBNAct(feat_ch, feat_ch, 3),
        )

    def forward(self, feat, coarse_logits, edge_logits, topo_logits):
        coarse_prob = torch.sigmoid(coarse_logits)
        edge_prob = torch.sigmoid(edge_logits)
        topo_prob = torch.sigmoid(topo_logits)
        uncertainty = 4.0 * coarse_prob * (1.0 - coarse_prob)
        merged = torch.cat([feat, coarse_prob, edge_prob, topo_prob, uncertainty], dim=1)
        return self.mix(merged)


class TopologyHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, in_ch, 3),
            ConvBNAct(in_ch, in_ch // 2, 3),
            nn.Conv2d(in_ch // 2, 1, 1),
        )

    def forward(self, x):
        return self.block(x)


class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, depth, stride, dilations, drop_paths):
        super().__init__()
        self.down = ConvBNAct(in_ch, out_ch, 3, stride=stride)
        blocks = []
        for i in range(depth):
            dilation = dilations[i % len(dilations)]
            blocks.append(ResidualDWBlock(out_ch, dilation=dilation, drop_path=drop_paths[i]))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.down(x)
        return self.blocks(x)


@dataclass
class SolarTopoCrackV2P1Config:
    in_channels: int = 1
    num_classes: int = 1
    base_channels: int = 32
    depths: Tuple[int, int, int, int] = (2, 2, 4, 2)
    drop_path_rate: float = 0.1
    deep_supervision: bool = True
    spectral_cutoff: float = 0.16


class SolarTopoCrackV2P1Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        chs = [cfg.base_channels, cfg.base_channels * 2, cfg.base_channels * 4, cfg.base_channels * 8]
        total_blocks = sum(cfg.depths)
        dpr = torch.linspace(0, cfg.drop_path_rate, total_blocks).tolist()

        self.stem = nn.Sequential(
            ConvBNAct(cfg.in_channels, chs[0], 3, stride=1),
            ConvBNAct(chs[0], chs[0], 3, stride=1),
        )

        cursor = 0
        self.stage1 = EncoderStage(chs[0], chs[0], cfg.depths[0], stride=2,
                                   dilations=(1, 2), drop_paths=dpr[cursor: cursor + cfg.depths[0]])
        cursor += cfg.depths[0]
        self.stage2 = EncoderStage(chs[0], chs[1], cfg.depths[1], stride=2,
                                   dilations=(1, 2), drop_paths=dpr[cursor: cursor + cfg.depths[1]])
        cursor += cfg.depths[1]
        self.stage3 = EncoderStage(chs[1], chs[2], cfg.depths[2], stride=2,
                                   dilations=(1, 2), drop_paths=dpr[cursor: cursor + cfg.depths[2]])
        cursor += cfg.depths[2]
        self.stage4 = EncoderStage(chs[2], chs[3], cfg.depths[3], stride=2,
                                   dilations=(1, 2), drop_paths=dpr[cursor: cursor + cfg.depths[3]])

        self.edge_pyramid = EdgePyramid(cfg.in_channels, feat_channels=chs)
        self.spectral_pyramid = SpectralPyramid(feat_channels=chs, cutoff=cfg.spectral_cutoff)
        self.context = MultiScaleContext(chs[3])
        self.orientation = OrientationAwareMixer(chs[3], drop_path=cfg.drop_path_rate)

        self.dec3 = DecoderBlock(chs[3], chs[2], chs[2], edge_ch=chs[2], spec_ch=chs[2])
        self.dec2 = DecoderBlock(chs[2], chs[1], chs[1], edge_ch=chs[1], spec_ch=chs[1])
        self.dec1 = DecoderBlock(chs[1], chs[0], chs[0], edge_ch=chs[0], spec_ch=chs[0])

        self.pre_refine = nn.Sequential(
            ConvBNAct(chs[0], chs[0], 3),
            ResidualDWBlock(chs[0], dilation=1),
        )
        out_ch = cfg.num_classes if cfg.num_classes > 1 else 1
        self.coarse_head = nn.Conv2d(chs[0], out_ch, 1)
        self.edge_head = nn.Sequential(
            ConvBNAct(chs[0], chs[0] // 2, 3),
            nn.Conv2d(chs[0] // 2, 1, 1),
        )
        self.topo_head = TopologyHead(chs[0])
        self.boundary_topo_refine = BoundaryTopologyRefiner(chs[0])
        self.final_head = nn.Sequential(
            ConvBNAct(chs[0], chs[0], 3),
            nn.Conv2d(chs[0], out_ch, 1),
        )

        if cfg.deep_supervision:
            self.aux3 = nn.Conv2d(chs[2], out_ch, 1)
            self.aux2 = nn.Conv2d(chs[1], out_ch, 1)
            self.aux1 = nn.Conv2d(chs[0], out_ch, 1)
        else:
            self.aux3 = None
            self.aux2 = None
            self.aux1 = None

    def forward(self, x):
        x0 = self.stem(x)
        e1 = self.stage1(x0)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)

        edge_feats = self.edge_pyramid(x)
        spec_feats = self.spectral_pyramid(x)

        b = self.context(e4)
        b = self.orientation(b)

        d3 = self.dec3(b, e3, edge_feats[2], spec_feats[2])
        d2 = self.dec2(d3, e2, edge_feats[1], spec_feats[1])
        d1 = self.dec1(d2, e1, edge_feats[0], spec_feats[0])
        d0 = F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.pre_refine(d0)

        coarse_logits = self.coarse_head(d0)
        edge_logits = self.edge_head(d0)
        topo_logits = self.topo_head(d0)
        d0 = self.boundary_topo_refine(d0, coarse_logits[:, :1], edge_logits, topo_logits)
        logits = self.final_head(d0)

        outputs = {
            "logits": logits,
            "edge_logits": edge_logits,
            "topo_logits": topo_logits,
            "coarse_logits": coarse_logits,
        }
        if self.cfg.deep_supervision:
            aux_logits = [
                F.interpolate(self.aux3(d3), size=x.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(self.aux2(d2), size=x.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(self.aux1(d1), size=x.shape[-2:], mode="bilinear", align_corners=False),
            ]
            outputs["aux_logits"] = aux_logits
        return outputs
