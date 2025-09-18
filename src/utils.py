import logging
from datetime import datetime
import os
import random
import numpy as np
import torch
def setup_logging(father_folder_name, folder_name):
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    log_file_path = os.path.join(father_folder_name, folder_name, "log.txt")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def get_timestamp():
    timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    return timestamp

def create_save_folders(mode, dataset, num_experts, moe_model, save_root):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{mode}_{dataset}_{num_experts}experts_{moe_model}_{timestamp}"
    os.makedirs(save_root, exist_ok=True)
    folder_path = os.path.join(save_root, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return save_root, folder_name

def delete_model(father, folder, min_turn):
    for f in os.listdir(os.path.join(father, folder)):
        if not (f == f"checkpoint_{min_turn}_epoch.pkl" or f == "log.txt"):
            os.remove(os.path.join(father, folder, f))

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False