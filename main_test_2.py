from multiprocessing import freeze_support
import argparse


 
config_dict = None

if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="This program trains and tests a deep " + "learning model to regress on CDS data")
    parser.add_argument("--config", dest="config", help="Set path to config file.")
    args = parser.parse_args()

if args.config:
    config_dict = config.get_args(args.config)
    # test()
    # train()
    #else:
    #raise ValueError("Config file not set. Use '--config <path_to_file>' to load a configuration.")

