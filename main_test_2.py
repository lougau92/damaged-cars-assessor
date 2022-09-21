from calendar import c
from multiprocessing import freeze_support
import argparse
from config import Config
from train_ae import train_CAE
config_dict = None
from utils import parse_transforms
from classification import train_model

if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="This program trains and tests a deep " + "learning model to regress on CDS data")
    parser.add_argument("--config", dest="config", help="Set path to config file.")
    args = parser.parse_args()

if args.config:
    config_dict = Config.get_args(path = args.config)

    if config_dict["task"] == "classification":
        train_model(
            data_path =  config_dict["data_path"],
            lr = config_dict["lr"],
            model_type = config_dict["model_type"],
            transforms= parse_transforms(config_dict["transforms"].split(','),config_dict["img_size"]),
            classes= config_dict["classes"],
            config_path= args.config
        )
    else :
        train_CAE(
            data_root= config_dict["dataset_path"],
            img_size= config_dict["img_size"],
            latent_dim= config_dict["latent_dim"],
            lr = config_dict["lr"],
            epochs = config_dict["epochs"],
            # train_batch= config_dict["train_batch"],
            transforms= parse_transforms(config_dict["transforms"].split(','),config_dict["img_size"]),
            config_path= args.config,
            # patience=config_dict["patience"],
            # min_lr=config_dict["min_lr"],
            # factor=config_dict["factor"],
            # mode=config_dict["mode"],
            dataset=config_dict["dataset"],
            checkpoint_path= config_dict["checkpoint_path"],
            # model_class = config_dict["model_class"]
        )
    
else:
    raise ValueError("Config file not set. Use '--config <path_to_file>' to load a configuration.")

