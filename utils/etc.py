"""Module for utilities"""
import os

def check_path(path):
    """Function to check and make path"""
    if not os.path.exists(path):
        os.mkdir(path)
