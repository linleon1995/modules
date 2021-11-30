import os
import sys
import logging
import importlib
import random
import numpy as np
import torch
import torch.nn as nn  
import torch.optim as optim


def calculate_display_step(num_sample, batch_size, display_times=5):
    num_steps = max(num_sample//batch_size, 1)
    display_steps = max(num_steps//display_times//display_times*display_times, 1)
    return display_steps


def minmax_norm(data):
    data_shape = data.size()
    data = data.view(data.size(0), -1)
    mins = data.min(1, keepdim=True)[0]
    maxs = data.max(1, keepdim=True)[0]
    data = (data-mins) / (maxs-mins)
    data = data.view(data_shape)
    return data


def set_deterministic(manual_seed, python_lib, numpy_lib, pytorch_lib):
    # see https://pytorch.org/docs/stable/notes/randomness.html
    if python_lib:
        random.seed(manual_seed)

    if numpy_lib:
        np.random.seed(manual_seed)

    if pytorch_lib:
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def create_training_path(train_logdir):
    idx = 0
    path = os.path.join(train_logdir, "run_{:03d}".format(idx))
    while os.path.exists(path):
        # if len(os.listdir(path)) == 0:
        #     os.remove(path)
        idx += 1
        path = os.path.join(train_logdir, "run_{:03d}".format(idx))
    os.makedirs(path)
    return path


loggers = {}
def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # TODO: Understand why "logger.propagate = False" can prevent duplicate logging inforamtion
        logger.propagate = False
        loggers[name] = logger
        return logger


# TODO: different indent of dataset config, preprocess config, train config
# TODO: recursively
def config_logging(path, config, access_mode):
    with open(path, access_mode) as fw:
        for dict_key in config:
            dict_value = config[dict_key]
            if isinstance(dict_value , dict):
                for sub_dict_key in dict_value:
                    fw.write(f'{sub_dict_key}: {dict_value[sub_dict_key]}\n')
            else:
                fw.write(f'{dict_key}: {dict_value}\n')


def create_optimizer(optimizer_config, model):
    # TODO: add SGD
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0)
    momentum = optimizer_config.get('momentum', 0)
    betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    optimizer_name = optimizer_config['optimizer']
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, betas=betas, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError('Unknown optimizer name.')
    return optimizer


def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config['optimizer'] = optimizer
    return clazz(**lr_config)


def create_criterion(name):
    if name == 'CrossEntropy':
        loss_func = nn.CrossEntropyLoss()
    elif name == 'BCE':
        loss_func = nn.BCEWithLogitsLoss()
    else:
        raise ValueError('Unknown Loss name.')
    return loss_func


def create_activation(name):
    if name == 'sigmoid':
        activation = torch.sigmoid()
    elif name == 'softmax':
        activation = torch.nn.functional.softmax()
    else:
        raise ValueError('Unknown Loss name.')
    return activation