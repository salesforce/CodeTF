import os
import sys
from pathlib import Path
# sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
from omegaconf import OmegaConf
from codetf.common.registry import registry

root_dir = os.path.dirname(os.path.abspath(__file__))
default_config = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
registry.register_path("repo_root", os.path.join(root_dir, ".."))
registry.register_path("cache_root", default_config.env.cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])
