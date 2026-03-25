import random 
import numpy as np
import torch 
import yaml 
from pathlib import Path 

def set_seed(seed: int = 42):
    random.seed()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.batch_norm_elemt.cudnn.benchmark = False
    
def load_config(config_path:str) ->dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def save_config(config: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)