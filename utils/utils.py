import math
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def show_network(args, model, model_name):
    parameters_num = sum(p.numel() for p in model.parameters()
                         if p.requires_grad)  # p.numel()返回张量中元素的个数
    # args.logger.debug(model)
    args.logger.debug(
        f'[*] The {model_name} has {parameters_num:,} trainable parameters!')


def get_trainable_parameters(model, skip_list=[]):
    trainable_parameters = []
    for name, param in model.named_parameters():
        if not name.split('.')[0] in skip_list:
            if param.requires_grad:
                trainable_parameters.append(param)
    return trainable_parameters


def show_running_status(args):
    welcome_info = f'[*] model_name: {args.model_name}, run_id: {args.run_id}, ckpt_id: {args.ckpt_id}, mode: {args.mode}'
    args.logger.info(welcome_info)


def set_random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True


def save_args_to_yaml(args):
    arguments = ''
    for key, value in dict(sorted(args.__dict__.items())).items():
        if isinstance(value, (str, int, float)):
            arguments += f'{key}: {value}\n'
    with open(args.args_path, 'w') as f:
        f.write(arguments.strip())


def load_args_from_yaml(args, yaml_path, skip_list):
    with open(yaml_path, 'r') as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in args_dict.items():
            if key not in skip_list:
                setattr(args, key, value)
    return args


def show_running_time(args, start_time, end_time):
    running_s = math.ceil(end_time - start_time)
    running_m = running_s // 60
    running_s = running_s % 60
    running_h = running_m // 60
    running_m = running_m % 60
    running_time = f'{running_h:02}h-{running_m:02}m-{running_s:02}s'
    args.logger.debug(
        f'[*] The running time of this experiment is {running_time}!\n')


def get_time():
    current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    return current_time


def convert_dict_to_string(loss_dict):
    loss_list = list(map(lambda x: f'{x[0]}: {x[1]:.3f}', loss_dict.items()))
    return ' | '.join(loss_list)
