from models.solar_edge_msf import SolarEdgeMSFConfig, SolarEdgeMSFNet
from models.solar_spec_crack import SolarSpecCrackConfig, SolarSpecCrackNet
from models.solar_topo_crack import SolarTopoCrackConfig, SolarTopoCrackNet
from models.solar_topo_crack_v2p1 import SolarTopoCrackV2P1Config, SolarTopoCrackV2P1Net


def build_model(config):
    model_name = str(config.get("model", {}).get("name", "SolarTopoCrackNet"))

    if model_name == "SolarEdgeMSFNet":
        model_cfg = SolarEdgeMSFConfig(
            in_channels=int(config["model"].get("in_channels", 1)),
            num_classes=int(config["model"].get("num_classes", 1)),
            base_channels=int(config["model"].get("base_channels", 32)),
            depths=tuple(config["model"].get("depths", [2, 2, 4, 2])),
            drop_path_rate=float(config["model"].get("drop_path_rate", 0.1)),
            deep_supervision=bool(config["model"].get("deep_supervision", True)),
        )
        return SolarEdgeMSFNet(model_cfg)

    if model_name == "SolarSpecCrackNet":
        model_cfg = SolarSpecCrackConfig(
            in_channels=int(config["model"].get("in_channels", 1)),
            num_classes=int(config["model"].get("num_classes", 1)),
            base_channels=int(config["model"].get("base_channels", 32)),
            depths=tuple(config["model"].get("depths", [2, 2, 4, 2])),
            drop_path_rate=float(config["model"].get("drop_path_rate", 0.1)),
            deep_supervision=bool(config["model"].get("deep_supervision", True)),
            spectral_cutoff=float(config["model"].get("spectral_cutoff", 0.16)),
        )
        return SolarSpecCrackNet(model_cfg)


    if model_name == "SolarTopoCrackV2P1Net":
        model_cfg = SolarTopoCrackV2P1Config(
            in_channels=int(config["model"].get("in_channels", 1)),
            num_classes=int(config["model"].get("num_classes", 1)),
            base_channels=int(config["model"].get("base_channels", 32)),
            depths=tuple(config["model"].get("depths", [2, 2, 4, 2])),
            drop_path_rate=float(config["model"].get("drop_path_rate", 0.1)),
            deep_supervision=bool(config["model"].get("deep_supervision", True)),
            spectral_cutoff=float(config["model"].get("spectral_cutoff", 0.16)),
        )
        return SolarTopoCrackV2P1Net(model_cfg)

    if model_name == "SolarTopoCrackNet":
        model_cfg = SolarTopoCrackConfig(
            in_channels=int(config["model"].get("in_channels", 1)),
            num_classes=int(config["model"].get("num_classes", 1)),
            base_channels=int(config["model"].get("base_channels", 32)),
            depths=tuple(config["model"].get("depths", [2, 2, 4, 2])),
            drop_path_rate=float(config["model"].get("drop_path_rate", 0.1)),
            deep_supervision=bool(config["model"].get("deep_supervision", True)),
            spectral_cutoff=float(config["model"].get("spectral_cutoff", 0.16)),
        )
        return SolarTopoCrackNet(model_cfg)

    raise ValueError("Unsupported model name: {}".format(model_name))
