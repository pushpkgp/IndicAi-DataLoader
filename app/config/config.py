import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def get_path(self, key):
        return Path(self.cfg["paths"].get(key))

    def get(self, *keys, default=None):
        val = self.cfg
        for key in keys:
            val = val.get(key, {})
        return val if val else default