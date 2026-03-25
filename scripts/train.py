import sys 
from pathlib import Path 

sys.path.insert(0,str(Path(__file__).parent.parent/'src'))

import argparse 
from utils import load_config, set_seed

def main(config_path: str):
    config = load_config(config_path)
    set_seed = config['seed']
    
    print(f"Конфиг:{config}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segmentation road model")
    parser.add_argument('--config',type=str, default='configs/config.yaml', help = "Path to config file")
    args = parser.parse_args()
    main(args.config)

