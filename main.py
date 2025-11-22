import torch
import numpy as np
from sys import argv
import json
from scripts.time_log import time_log_module as tlm
from scripts.logger import logger
from data.data import data
from model.model import tokenizer
from model.model import embedding
from model.model import model

# Configuration
with open("config.json", "r") as f:
    config = json.load(f)
webhook_url = config.get("webhook_url", "") # leave empty to disable webhook logging
model_path = config.get("model_path", "model/model.pth")
data_path = config.get("data_path", "data/tiny_sheakespeare.txt") # Tiny Shakespeare Dataset by default
version = config.get("version", "None")
dataset_loading_size = config.get("dataset_loading_size", 10000)
tokenizer_config = config.get("tokenizer", {})
embedding_config = config.get("embedding", {})
del config

# Args
train = "--train" in argv or "-t" in argv
predict = "--predict" in argv
chat - = "--chat" in argv
force_cpu = "--cpu" in sys.argv
force_cuda = "--cuda" in sys.argv

if __name__ == "__main__":
    print(f"{tlm()} Start of program.")
    logger = logger(discord_webhook=webhook_url) # créer le logger

    # Logging system info
    logger.log(f"Lookmaxx.ai V{version}.", v=True, Wh=True, mention=False)
    logger.log(f"To change any setting, go check config.json.", v=True, Wh=True, mention=False)
    logger.log(f"PyTorch version: {torch.__version__}", v=True, Wh=True, mention=False)
    logger.log(f"CUDA status : {str(torch.cuda.is_available())}", v=True, Wh=True, mention=False)
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        msg = f"{count} GPU{'s' if count > 1 else ''} detected."
        logger.log(msg, v=True, Wh=True, mention=False)
        for i in range(count):
            logger.log(f" -> Device {i}: {torch.cuda.get_device_name(i)}", v=True, Wh=True, mention=False)
    
    # Select device
    if force_cuda:
        if not torch.cuda.is_available(): # Si pas de GPU
            logger.log("CUDA forced but no GPU detected. Exiting.", v=False, Wh=True, mention=True)
            raise EnvironmentError(f"{tlm()} CUDA forced but no GPU detected. Exiting.")
        if force_cpu: # Si les deux sont forcés
            logger.log("Both --force-cuda and --force-cpu flags detected. Please choose only one.", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Both --force-cuda and --force-cpu flags detected. Please choose only one.")
        logger.log("CUDA forced. Using GPU for computations.", v=True, Wh=True, mention=False)
        device = torch.device("cuda")
    elif force_cpu:
        logger.log("CPU forced. Using CPU for computations.", v=True, Wh=True, mention=False)
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.log(f"Using device: {device}.", v=True, Wh=True, mention=False)
    
    if train:
        # Load data
        logger.log(f"Loading data from {data_path}...", v=True, Wh=True, mention=False)
        data_loader = data(logger, data_path, dataset_loading_size)
        dataset = data_loader.load_data()
        del data_loader
        
        # Initialize tokenizer
        logger.log("Initializing tokenizer...", v=True, Wh=True, mention=False)
        tk = tokenizer(logger, tokenizer_config)
        if not tk.vocab_status():
            logger.log("No existing tokenizer found. Creating new vocabulary...", v=True, Wh=True, mention=False)
            tk.create_vocab(dataset)
            tk.save_vocab()
        else:
            logger.log("Existing tokenizer found. Loading vocabulary...", v=True, Wh=True, mention=False)
            tk.load_vocab()
        del dataset # Free memory

    if predict:
        pass
    
    if chat:
        pass