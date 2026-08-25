from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
                 padding: Optional[int] = None, groups: int = 1, act: bool = True) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDWBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, drop_path: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 7, padding=3 * dilation,
                               dilation=dilation, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels * 4, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(channels * 4, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.bn2(x)
        x = identity + self.drop_path(x)
        return self.act(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class FixedSobel(nn.Module):
    """Fixed Sobel edge extractor for thin crack / boundary emphasis."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("gx", gx.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.register_buffer("gy", gy.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.gx, padding=1, groups=self.in_channels)
        gy = F.conv2d(x, self.gy, padding=1, groups=self.in_channels)
        mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        return mag


class EdgePyramid(nn.Module):
    def __init__(self, in_channels: int, feat_channels: List[int]) -> None:
        super().__init__()
        self.sobel = FixedSobel(in_channels)
        self.proj = nn.ModuleList()
        prev = in_channels
        for out_ch in feat_channels:
            self.proj.append(
                nn.Sequential(
                    ConvBNAct(prev, out_ch, 3, stride=2),
                    ConvBNAct(out_ch, out_ch, 3, stride=1),
                )
            )
            prev = out_ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        edge = self.sobel(x)
        outs = []
        cur = edge
        for block in self.proj:
            cur = block(cur)
            outs.append(cur)
        return outs


class SpatialGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class EdgeGuidedFusion(nn.Module):
    def __init__(self, feat_ch: int, edge_ch: int) -> None:
        super().__init__()
        self.edge_proj = ConvBNAct(edge_ch, feat_ch, 1)
        self.mix = ConvBNAct(feat_ch * 2, feat_ch, 3)
        self.gate = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        edge_feat = F.interpolate(edge_feat, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        edge_feat = self.edge_proj(edge_feat)
        merged = torch.cat([feat, edge_feat], dim=1)
        gate = self.gate(merged)
        fused = feat * (1.0 - gate) + edge_feat * gate
        fused = torch.cat([fused, feat], dim=1)
        return self.mix(fused)


class MultiScaleContext(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        dilations = [1, 2, 4, 6]
        self.branches = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv(channels, channels // 2, dilation=d),
                SpatialGate(channels // 2),
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
            ConvBNAct(channels, channels, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(gp)
        return self.fuse(torch.cat(feats, dim=1))


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int, stride: int,
                 dilations: Tuple[int, ...], drop_paths: List[float]) -> None:
        super().__init__()
        self.down = ConvBNAct(in_ch, out_ch, 3, stride=stride)
        blocks = []
        for i in range(depth):
            dilation = dilations[i % len(dilations)]
            blocks.append(ResidualDWBlock(out_ch, dilation=dilation, drop_path=drop_paths[i]))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.blocks(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, edge_ch: int) -> None:
        super().__init__()
        self.up_proj = ConvBNAct(in_ch, out_ch, 1)
        self.skip_proj = ConvBNAct(skip_ch, out_ch, 1)
        self.edge_fuse = EdgeGuidedFusion(out_ch, edge_ch)
        self.refine = nn.Sequential(
            ConvBNAct(out_ch * 2, out_ch, 3),
            ResidualDWBlock(out_ch, dilation=1),
            ResidualDWBlock(out_ch, dilation=2),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_proj(x)
        skip = self.skip_proj(skip)
        skip = self.edge_fuse(skip, edge_feat)
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)


@dataclass
class SolarEdgeMSFConfig:
    in_channels: int = 1
    num_classes: int = 1
    base_channels: int = 32
    depths: Tuple[int, int, int, int] = (2, 2, 4, 2)
    drop_path_rate: float = 0.1
    deep_supervision: bool = True


class SolarEdgeMSFNet(nn.Module):
    """
    SolarEdgeMSFNet
    ----------------
    A defect-segmentation network designed for EL solar-cell imagery.

    Main ideas:
      1. fixed Sobel edge pyramid for crack / boundary emphasis
      2. multi-scale context bottleneck for low-contrast defects
      3. edge-guided decoder fusion to preserve tiny structures
      4. optional deep supervision and auxiliary edge head
    """

    def __init__(self, cfg: SolarEdgeMSFConfig) -> None:
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
                                   dilations=(1, 2, 4), drop_paths=dpr[cursor: cursor + cfg.depths[2]])
        cursor += cfg.depths[2]
        self.stage4 = EncoderStage(chs[2], chs[3], cfg.depths[3], stride=2,
                                   dilations=(1, 2, 4), drop_paths=dpr[cursor: cursor + cfg.depths[3]])

        self.edge_pyramid = EdgePyramid(cfg.in_channels, feat_channels=chs)
        self.context = MultiScaleContext(chs[3])

        self.dec3 = DecoderBlock(chs[3], chs[2], chs[2], edge_ch=chs[2])
        self.dec2 = DecoderBlock(chs[2], chs[1], chs[1], edge_ch=chs[1])
        self.dec1 = DecoderBlock(chs[1], chs[0], chs[0], edge_ch=chs[0])

        self.final_refine = nn.Sequential(
            ConvBNAct(chs[0], chs[0], 3),
            ResidualDWBlock(chs[0], dilation=1),
        )

        out_ch = cfg.num_classes if cfg.num_classes > 1 else 1
        self.seg_head = nn.Conv2d(chs[0], out_ch, 1)
        self.edge_head = nn.Sequential(
            ConvBNAct(chs[0], chs[0] // 2, 3),
            nn.Conv2d(chs[0] // 2, 1, 1)
        )

        if cfg.deep_supervision:
            self.aux3 = nn.Conv2d(chs[2], out_ch, 1)
            self.aux2 = nn.Conv2d(chs[1], out_ch, 1)
            self.aux1 = nn.Conv2d(chs[0], out_ch, 1)
        else:
            self.aux3 = None
            self.aux2 = None
            self.aux1 = None

    def forward(self, x: torch.Tensor):
        x0 = self.stem(x)
        e1 = self.stage1(x0)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)

        edge_feats = self.edge_pyramid(x)
        b = self.context(e4)

        d3 = self.dec3(b, e3, edge_feats[2])
        d2 = self.dec2(d3, e2, edge_feats[1])
        d1 = self.dec1(d2, e1, edge_feats[0])
        d0 = F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.final_refine(d0)

        seg = self.seg_head(d0)
        edge = self.edge_head(d0)

        if not self.cfg.deep_supervision:
            return {"logits": seg, "edge_logits": edge}

        aux1 = self.aux1(d1)
        aux2 = self.aux2(d2)
        aux3 = self.aux3(d3)
        aux1 = F.interpolate(aux1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        aux2 = F.interpolate(aux2, size=x.shape[-2:], mode="bilinear", align_corners=False)
        aux3 = F.interpolate(aux3, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {
            "logits": seg,
            "edge_logits": edge,
            "aux_logits": [aux1, aux2, aux3],
        }


def build_model(config: dict) -> SolarEdgeMSFNet:
    model_cfg = SolarEdgeMSFConfig(
        in_channels=int(config["model"].get("in_channels", 1)),
        num_classes=int(config["model"].get("num_classes", 1)),
        base_channels=int(config["model"].get("base_channels", 32)),
        depths=tuple(config["model"].get("depths", [2, 2, 4, 2])),
        drop_path_rate=float(config["model"].get("drop_path_rate", 0.1)),
        deep_supervision=bool(config["model"].get("deep_supervision", True)),
    )
    return SolarEdgeMSFNet(model_cfg)
