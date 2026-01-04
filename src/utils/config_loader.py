import yaml 
import os

def load_config(path: str = "config/config.yaml") -> dict:

    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found at : {path}")
    
    with open(path, "r") as f:
        return yaml.safe_load(f)