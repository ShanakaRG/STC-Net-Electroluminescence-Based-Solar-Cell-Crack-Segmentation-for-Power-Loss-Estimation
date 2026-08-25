import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class CSVLogger:
    def __init__(self, filepath: Union[str, Path], fieldnames) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self.initialized = self.filepath.exists()

    def log(self, row: Dict[str, Any]) -> None:
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self.initialized:
                writer.writeheader()
                self.initialized = True
            writer.writerow(row)


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(state: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def state_dict_for_saving(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def flexible_load_model(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    stripped = {}
    has_module_prefix = False
    for k, v in state_dict.items():
        if k.startswith("module."):
            stripped[k[7:]] = v
            has_module_prefix = True
        else:
            stripped[k] = v
    if has_module_prefix:
        model.load_state_dict(stripped)
        return

    prefixed = {}
    for k, v in state_dict.items():
        prefixed["module." + k] = v
    model.load_state_dict(prefixed)
