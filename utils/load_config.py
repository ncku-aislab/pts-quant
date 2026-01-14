import argparse
import os
import yaml

def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

def load_config(config_file):
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook(config_file=config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--deepfake", type=str, default=None)
    parser.add_argument("--exp", type=int, default=None)
    args = parser.parse_args([])
    return args
