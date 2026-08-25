from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dims)
        union = probs.sum(dims) + targets.sum(dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def soft_erode(img):
    if img.ndim != 4:
        return img
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), stride=1, padding=1)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iters=10):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iters=10, smooth=1e-6):
        super().__init__()
        self.iters = int(iters)
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        skel_p = soft_skel(probs, self.iters)
        skel_t = soft_skel(targets, self.iters)
        tprec = (skel_p * targets).sum(dim=(1,2,3)) / (skel_p.sum(dim=(1,2,3)) + self.smooth)
        tsens = (skel_t * probs).sum(dim=(1,2,3)) / (skel_t.sum(dim=(1,2,3)) + self.smooth)
        cl = 1.0 - (2.0 * tprec * tsens + self.smooth) / (tprec + tsens + self.smooth)
        return cl.mean()


class BoundaryAwareBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=1.0, edge_boost=2.0, focal_gamma=0.0, ohem_ratio=1.0, ohem_min_kept=4096):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.edge_boost = float(edge_boost)
        self.focal_gamma = float(focal_gamma)
        self.ohem_ratio = float(ohem_ratio)
        self.ohem_min_kept = int(ohem_min_kept)
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.dice = DiceLoss()

    def _ohem(self, loss_map):
        if self.ohem_ratio >= 0.999:
            return loss_map.mean()
        flat = loss_map.view(loss_map.shape[0], -1)
        k = max(self.ohem_min_kept, int(flat.shape[1] * self.ohem_ratio))
        k = min(k, flat.shape[1])
        topk = torch.topk(flat, k=k, dim=1, sorted=False)[0]
        return topk.mean()

    def forward(self, logits, targets, edge_targets=None):
        targets = targets.float()
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        pos_weight = self.pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
        if self.focal_gamma > 0:
            prob = torch.sigmoid(logits)
            pt = prob * targets + (1.0 - prob) * (1.0 - targets)
            bce = bce * torch.pow(1.0 - pt, self.focal_gamma)
        if edge_targets is not None:
            if edge_targets.ndim == 3:
                edge_targets = edge_targets.unsqueeze(1)
            bce = bce * (1.0 + self.edge_boost * edge_targets.float())
        bce = self._ohem(bce)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class MultiClassCEDice(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super().__init__()
        weight_tensor = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("weight_tensor", weight_tensor)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice = MultiClassDiceLoss()

    def forward(self, logits, targets):
        weight_tensor = self.weight_tensor
        if weight_tensor is not None:
            weight_tensor = weight_tensor.to(logits.device)
        ce = F.cross_entropy(logits, targets.long(), weight=weight_tensor)
        dice = self.dice(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class EdgeLoss(nn.Module):
    def __init__(self, pos_weight=2.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits, targets):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=self.pos_weight.to(logits.device))


class TopologyLoss(nn.Module):
    def __init__(self, bce_weight=0.4, dice_weight=0.4, cldice_weight=0.2, pos_weight=4.0, cldice_iters=10):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.cldice_weight = float(cldice_weight)
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.dice = DiceLoss()
        self.cldice = SoftCLDiceLoss(iters=cldice_iters)

    def forward(self, logits, targets):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=self.pos_weight.to(logits.device))
        dice = self.dice(logits, targets)
        cl = self.cldice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice + self.cldice_weight * cl


class TopologyConsistencyLoss(nn.Module):
    def __init__(self, boundary_weight=0.5, inside_weight=1.0, anti_edge_weight=0.25):
        super().__init__()
        self.boundary_weight = float(boundary_weight)
        self.inside_weight = float(inside_weight)
        self.anti_edge_weight = float(anti_edge_weight)

    def _soft_boundary(self, mask_prob):
        dil = F.max_pool2d(mask_prob, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool2d(-mask_prob, kernel_size=3, stride=1, padding=1)
        return torch.clamp(dil - ero, min=0.0, max=1.0)

    def forward(self, mask_logits, edge_logits, topo_logits):
        mask_prob = torch.sigmoid(mask_logits[:, :1])
        edge_prob = torch.sigmoid(edge_logits)
        topo_prob = torch.sigmoid(topo_logits)
        soft_boundary = self._soft_boundary(mask_prob)
        boundary_term = F.l1_loss(edge_prob, soft_boundary)
        inside_term = ((1.0 - mask_prob) * topo_prob).mean()
        anti_edge_term = (edge_prob * topo_prob).mean()
        return self.boundary_weight * boundary_term + self.inside_weight * inside_term + self.anti_edge_weight * anti_edge_term


class CombinedSolarLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_classes = int(cfg["model"].get("num_classes", 1))
        loss_cfg = cfg["loss"]
        self.seg_weight = float(loss_cfg.get("seg_weight", 1.0))
        self.edge_weight = float(loss_cfg.get("edge_weight", 0.2))
        self.topo_weight_full = float(loss_cfg.get("topo_weight", 0.2))
        self.consistency_weight_full = float(loss_cfg.get("consistency_weight", 0.1))
        self.aux_weight = float(loss_cfg.get("aux_weight", 0.3))
        self.coarse_weight = float(loss_cfg.get("coarse_weight", 0.25))
        self.topo_warmup_epochs = int(loss_cfg.get("topo_warmup_epochs", 8))
        self.current_epoch = 1
        if num_classes <= 1:
            self.seg_loss = BoundaryAwareBCEDiceLoss(
                bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
                dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
                pos_weight=float(loss_cfg.get("seg_pos_weight", 1.0)),
                edge_boost=float(loss_cfg.get("boundary_boost", 2.0)),
                focal_gamma=float(loss_cfg.get("focal_gamma", 0.0)),
                ohem_ratio=float(loss_cfg.get("ohem_ratio", 1.0)),
                ohem_min_kept=int(loss_cfg.get("ohem_min_kept", 4096)),
            )
        else:
            self.seg_loss = MultiClassCEDice(
                ce_weight=float(loss_cfg.get("ce_weight", 0.5)),
                dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
                class_weights=loss_cfg.get("class_weights"),
            )
        self.edge_loss = EdgeLoss(pos_weight=float(loss_cfg.get("edge_pos_weight", 2.0)))
        self.topo_loss = TopologyLoss(
            bce_weight=float(loss_cfg.get("topo_bce_weight", 0.4)),
            dice_weight=float(loss_cfg.get("topo_dice_weight", 0.4)),
            cldice_weight=float(loss_cfg.get("topo_cldice_weight", 0.2)),
            pos_weight=float(loss_cfg.get("topo_pos_weight", 4.0)),
            cldice_iters=int(loss_cfg.get("cldice_iters", 10)),
        )
        self.consistency_loss = TopologyConsistencyLoss(
            boundary_weight=float(loss_cfg.get("cons_boundary_weight", 0.5)),
            inside_weight=float(loss_cfg.get("cons_inside_weight", 1.0)),
            anti_edge_weight=float(loss_cfg.get("cons_anti_edge_weight", 0.25)),
        )
        self.num_classes = num_classes

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _warm(self):
        if self.topo_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(self.current_epoch) / float(self.topo_warmup_epochs))

    def _seg_term(self, logits, mask, edge):
        if self.num_classes <= 1:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            return self.seg_loss(logits, mask, edge)
        return self.seg_loss(logits, mask)

    def forward(self, outputs, batch):
        mask = batch["mask"]
        edge = batch["edge"]
        topo = batch.get("topo")
        warm = self._warm()
        seg = self._seg_term(outputs["logits"], mask, edge)
        edge_term = self.edge_loss(outputs["edge_logits"], edge)
        topo_total = torch.tensor(0.0, device=seg.device)
        if topo is not None and "topo_logits" in outputs:
            topo_total = self.topo_loss(outputs["topo_logits"], topo)
        consistency_total = torch.tensor(0.0, device=seg.device)
        if "topo_logits" in outputs:
            consistency_total = self.consistency_loss(outputs["logits"], outputs["edge_logits"], outputs["topo_logits"])
        aux_total = torch.tensor(0.0, device=seg.device)
        if "aux_logits" in outputs:
            aux_losses = [self._seg_term(aux, mask, edge) for aux in outputs["aux_logits"]]
            if aux_losses:
                aux_total = torch.stack(aux_losses).mean()
        coarse_total = torch.tensor(0.0, device=seg.device)
        if "coarse_logits" in outputs:
            coarse_total = self._seg_term(outputs["coarse_logits"], mask, edge)
        total = self.seg_weight * seg + self.edge_weight * edge_term + warm * (self.topo_weight_full * topo_total + self.consistency_weight_full * consistency_total) + self.aux_weight * aux_total + self.coarse_weight * coarse_total
        details = {"loss": total, "seg_loss": seg.detach(), "edge_loss": edge_term.detach(), "topo_loss": topo_total.detach(), "consistency_loss": consistency_total.detach(), "aux_loss": aux_total.detach(), "coarse_loss": coarse_total.detach()}
        return total, details
