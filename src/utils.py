import os
import torch
import torch.distributed as dist
import cfg
from collections import OrderedDict


def get_world_size():
    return dist.get_world_size()


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value


def init_distributed_mode():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(os.environ['LOCAL_RANK'])
    print(rank, world_size, gpu)
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=4, rank=rank)
    dist.barrier()


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_ckpt_dir(ckpt_dir):
    makedirs(ckpt_dir)
    tensorboard_dir = os.path.join(ckpt_dir, 'tensorboard_logs')
    save_dir = os.path.join(ckpt_dir, 'saved_models')
    makedirs(save_dir)
    makedirs(tensorboard_dir)
    if cfg.resume == False and cfg.finetune == False:
        del_file(tensorboard_dir)
    return save_dir, tensorboard_dir


def to_items(dic):
    return dict(map(_to_item, dic.items()))


def _to_item(item):
    return item[0], item[1].item()


def del_file(path_data):
    for i in os.listdir(path_data):  # 返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = os.path.join(path_data, i)  # 当前文件夹的下面的所有东西的绝对路径
        # 判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
        if os.path.isfile(file_data) == True:
            os.remove(file_data)
        else:
            del_file(file_data)


def fix_model_state_dict(state_dict, del_str='module.'):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith(del_str):
            name = name[len(del_str):]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict
