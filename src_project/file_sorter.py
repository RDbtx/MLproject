import os
import shutil


def file_mover(origin: str, destination: str):
    """A utility function that iters to all the file contained into
    the origin directory and moves them to the destination directory.
    This function was created to easily move files from one folder to another.
    Without any overhead of classic drag and drop.

    input:
    origin: path to the origin folder
    destination: path to the destination folder
    """
    for elem in os.listdir(origin):
        if "challenge" in elem:
            print(elem)
            shutil.move(origin + "/" + elem, destination + "/" + elem)
