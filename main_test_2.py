from multiprocessing import freeze_support
import argparse
from config import Config
from train_ae import train_CAE
config_dict = None
from utils import parse_transforms

if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="This program trains and tests a deep " + "learning model to regress on CDS data")
    parser.add_argument("--config", dest="config", help="Set path to config file.")
    args = parser.parse_args()

if args.config:
    config_dict = Config.get_args(path = args.config)

    train_CAE(
        data_root= config_dict["dataset_path"],
        img_size= config_dict["img_size"],
        latent_dim= config_dict["latent_dim"],
        lr = config_dict["lr"],
        epochs = config_dict["epochs"],
        train_batch= config_dict["train_batch"],
        transforms= parse_transforms(config_dict["transforms"].split(', '),config_dict["img_size"])
    )
    
else:
    raise ValueError("Config file not set. Use '--config <path_to_file>' to load a configuration.")

