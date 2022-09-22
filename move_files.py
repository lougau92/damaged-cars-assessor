import shutil
import os
import random
# moving files from a directory to another
def move_files (
    # dest = "~/projects/louis/cars_data/internet_dataset_cleaned/data1a/test/00-damage/",
    # source = "~/projects/louis/cars_data/internet_dataset_cleaned/data1a/training/00-damage/"
    path
    ):

    dest = path + "/test/01-whole/"
    source =path + "/training/01-whole/"

    list_imgs = os.listdir(source)

    # try:

    for i in range(50):
        test_set = random.sample(list_imgs,1)
        file = test_set[0]
        if file.endswith("JPEG") or file.endswith("jpeg") or file.endswith("jpg"):
            # construct full file path
            destination = dest + file
            # move file
            shutil.move(source+ file, destination)
            print("move",file)
    # except:
    #     print("error")
move_files(path = "/home/p63744//projects/louis/cars_data/internet_dataset_cleaned/data1a/")