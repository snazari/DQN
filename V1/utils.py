import os
# from config import config_saving


def make_dirs(data, model_name=None):
    if model_name is None:
        path = os.path.join("data", data)
    else:
        path = os.path.join("data", model_name, data)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
