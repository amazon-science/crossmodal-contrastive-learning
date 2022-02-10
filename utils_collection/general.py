import csv
import ctypes
import datetime
from easydict import EasyDict
import logging
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Union, Tuple, Dict
import yaml

from typing import Any, Dict, List, Tuple

import GPUtil
import numpy as np
import psutil
import torch as th
import torch
import torch.backends.cudnn as cudnn
from torch import cuda


EVALKEYS = ["r1", "r5", "r10", "r50", "medr", "meanr", "sum"]
EVALHEADER = "Retriev | R@1   | R@5   | R@10  | R@50  | MeanR |  MedR |    Sum"



def create_dataloader_path(data_root,
                           shot_per_group,
                           dataset_name,
                           text_feature_name='default',
                           feature_name_modality_a='action',
                           feature_name_modality_b='flow', pickle_path=None):
    """create the path to meta file and features
    #last modality will be modality_b

    Args:
        data_root ([PATH]): [Path to the data folder]
        shot_per_group ([Int]): [number of shots (clips) per group (video)]

    Returns:
        [Dict]: [path to meta data and video/language features]
    """

    meta_data_path = {}
    video_feat_path = {}

    if pickle_path is not None:
        pickle_path = Path(pickle_path)
    else:
        pickle_path = ""

    for mod_name in feature_name_modality_a:
        meta_data_path[mod_name] = Path(
        os.path.join(data_root, "meta",
                    "meta_group{}_{}.json".format(shot_per_group, mod_name)))

        video_feat_path[mod_name] = Path(
            os.path.join(data_root, "group{}".format(shot_per_group),
                        "video_features", "{}.h5".format(mod_name)))

    #If modality B is "text" then we already have it in language feats
    if feature_name_modality_b != "text":
        meta_data_path[feature_name_modality_b] = Path(
        os.path.join(data_root, "meta",
                        "meta_group{}_{}.json".format(shot_per_group, feature_name_modality_b)))

        video_feat_path[feature_name_modality_b] = Path(
                os.path.join(data_root, "group{}".format(shot_per_group),
                            "video_features", "{}.h5".format(feature_name_modality_b)))

    language_feat_path = Path(
        os.path.join(data_root, "group{}".format(shot_per_group),
                     "language_features",
                     "text_{}.h5".format(text_feature_name)))
    meta_text_len_path = Path(
        os.path.join(data_root, "group{}".format(shot_per_group),
                     "language_features",
                     "text_lens_{}.json".format(text_feature_name)))
    

    return {
        "meta_data": meta_data_path,
        "video_feats": video_feat_path,
        "language_feats": language_feat_path,
        "meta_text_len": meta_text_len_path,
        "dataset_name": dataset_name,
        "pickle_path": pickle_path
    }


def get_csv_header_keys(compute_clip_retrieval):
    metric_keys = ["ep", "time"]
    prefixes = ["v", "p"]
    if compute_clip_retrieval:
        prefixes += ["c", "s"]
    for prefix in prefixes:
        for key in EVALKEYS:
            metric_keys.append(f"{prefix}-{key}")
    return metric_keys


def print_csv_results(csv_file: str, cfg: EasyDict, print_fn=print):
    metric_keys = get_csv_header_keys(True)
    with Path(csv_file).open("rt", encoding="utf8") as fh:
        reader = csv.DictReader(fh, metric_keys)
        line_data = [line for line in reader][1:]
        for line in line_data:
            for key, val in line.items():
                line[key] = float(val)
    if cfg.val.det_best_field == "val_score_at_1":
        relevant_field = [line["v-r1"] + line["p-r1"] for line in line_data]
    elif cfg.val.det_best_field == "val_clip_score_at_1":
        relevant_field = [line["c-r1"] + line["s-r1"] for line in line_data]
    else:
        raise NotImplementedError
    best_epoch = np.argmax(relevant_field)

    def get_res(search_key):
        results = {}
        for key_, val_ in line_data[best_epoch].items():
            if key_[:2] == f"{search_key}-":
                results[key_[2:]] = float(val_)
        return results

    print_fn(f"Total epochs {len(line_data)}. "
             f"Results from best epoch {best_epoch}:")
    print_fn(EVALHEADER)
    print_fn(retrieval_results_to_str(get_res("p"), "Par2Vid"))
    print_fn(retrieval_results_to_str(get_res("v"), "Vid2Par"))
    print_fn(retrieval_results_to_str(get_res("s"), "Sen2Cli"))
    print_fn(retrieval_results_to_str(get_res("c"), "Cli2Sen"))


def expand_segment(num_frames, num_target_frames, start_frame, stop_frame):
    num_frames_seg = stop_frame - start_frame + 1
    changes = False
    if num_target_frames > num_frames:
        num_target_frames = num_frames
    if num_frames_seg < num_target_frames:
        while True:
            if start_frame > 0:
                start_frame -= 1
                num_frames_seg += 1
                changes = True
            if num_frames_seg == num_target_frames:
                break
            if stop_frame < num_frames - 1:
                stop_frame += 1
                num_frames_seg += 1
                changes = True
            if num_frames_seg == num_target_frames:
                break
    return start_frame, stop_frame, changes


def set_seed(seed: int, set_deterministic: bool = True):
    """
    Set all relevant seeds for torch, numpy and python

    Args:
        seed: int seed
        set_deterministic: Guarantee deterministic training, possibly at the cost of performance.
    """
    torch.manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if set_deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    elif cudnn.benchmark or not cudnn.deterministic:
        print("WARNING: Despite fixed seed {}, training may not be deterministic with {} "
              "(must be False for deterministic training) and {} (must be True for deterministic "
              "training)".format(seed, cudnn.benchmark, cudnn.deterministic))


def load_config(file: Union[str, Path]) -> EasyDict:
    with Path(file).open("rt", encoding="utf8") as fh:
        config = yaml.load(fh, Loader=yaml.Loader)
    cfg = EasyDict(config)
    # model symmetry
    for check_network in ["text_pooler", "text_sequencer"]:
        if getattr(cfg, check_network).name == "same":
            setattr(cfg, check_network,
                    getattr(cfg,
                            getattr(cfg, check_network).same_as))
    return cfg


def dump_config(cfg: EasyDict, file: Union[str, Path]) -> None:
    with Path(file).open("wt", encoding="utf8") as fh:
        yaml.dump(cfg, fh, Dumper=yaml.Dumper)


def print_config(cfg: EasyDict, level=0) -> None:
    for key, val in cfg.items():
        if isinstance(val, EasyDict):
            print("     " * level, str(key), sep="")
            print_config(val, level=level + 1)
        else:
            print("    " * level, f"{key} - f{val} ({type(val)})", sep="")


def make_shared_array(np_array: np.ndarray) -> mp.Array:
    flat_shape = int(np.prod(np_array.shape))
    shared_array_base = mp.Array(ctypes.c_float, flat_shape)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(np_array.shape)
    shared_array[:] = np_array[:]
    return shared_array


def compute_indices(num_frames_orig: int, num_frames_target: int,
                    is_train: bool):
    def round_half_down(array: np.ndarray) -> np.ndarray:
        return np.ceil(array - 0.5)

    if is_train:
        # random sampling during training
        start_points = np.linspace(0,
                                   num_frames_orig,
                                   num_frames_target,
                                   endpoint=False)
        start_points = round_half_down(start_points).astype(int)
        offsets = start_points[1:] - start_points[:-1]
        np.random.shuffle(offsets)
        last_offset = num_frames_orig - np.sum(offsets)
        offsets = np.concatenate([offsets, np.array([last_offset])])
        new_start_points = np.cumsum(offsets) - offsets[0]
        offsets = np.roll(offsets, -1)
        random_offsets = offsets * np.random.rand(num_frames_target)
        indices = new_start_points + random_offsets
        indices = np.floor(indices).astype(int)
        return indices
    # center sampling during validation
    start_points = np.linspace(0,
                               num_frames_orig,
                               num_frames_target,
                               endpoint=False)
    offset = num_frames_orig / num_frames_target / 2
    indices = start_points + offset
    indices = np.floor(indices).astype(int)
    return indices


def truncated_normal_fill(shape: Tuple[int],
                          mean: float = 0,
                          std: float = 1,
                          limit: float = 2) -> torch.Tensor:
    num_examples = 8
    tmp = torch.empty(shape + (num_examples, )).normal_()
    valid = (tmp < limit) & (tmp > -limit)
    _, ind = valid.max(-1, keepdim=True)
    return tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)


def retrieval_results_to_str(results: Dict[str, float], name: str):
    return ("{:7s} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:5.1f} | "
            "{:5.1f} | {:6.3f}").format(name, *[results[a] for a in EVALKEYS])


# def compute_retr_vid_to_par(video_feat, cap_feat):
#     similarity_scores = np.dot(video_feat, cap_feat.T)
#     return compute_retrieval_metrics(similarity_scores)

def compute_retr_vid_to_par(video_feat, cap_feat):
    num_points = video_feat.shape[0]
    d = np.dot(video_feat, cap_feat.T)
    return compute_retrieval_cosine(d, num_points)


def compute_retr_vid_to_par_softneighbor(video_feat, cap_feat):
    num_points = video_feat.shape[0]
    d = np.dot(video_feat, cap_feat.T)
    return compute_retrieval_softneighbor(d, num_points)


def compute_retr_par_to_vid_softneighbor(video_feat, cap_feat):
    num_points = video_feat.shape[0]
    d = np.dot(cap_feat, video_feat.T)
    return compute_retrieval_softneighbor(d, num_points)


def compute_retr_par_to_vid(video_feat, cap_feat):
    num_points = video_feat.shape[0]
    d = np.dot(cap_feat, video_feat.T)
    return compute_retrieval_cosine(d, num_points)

# def compute_retr_par_to_vid(video_feat, cap_feat):
#     similarity_scores = np.dot(cap_feat, video_feat.T)
#     return compute_retrieval_metrics(similarity_scores)

def compute_retrieval_coarse_to_fine(coarse_ind,  x_feat, y_feat):
    len_dot_product = x_feat.shape[0]
    dot_product = np.dot(x_feat, y_feat.T)
    ranks = np.zeros(len_dot_product)
    top1 = np.zeros(len_dot_product)
    ind_coarse_to_fine = []
    sum_corr = 0
    group_k = 10
    for index in range(len_dot_product):
        ind_coarse = index // group_k
        ind_fine = index - ind_coarse * group_k
        ind_h = coarse_ind[ind_coarse]
        if ind_h == ind_coarse:
            # print("correct")
            sum_corr += 1
            inds = np.argsort(dot_product[index, ind_coarse * group_k : (ind_coarse + 1) * group_k])[::-1]
        # print(inds, ind_fine)
            where = np.where(inds == ind_fine)
 
            rank = where[0][0]
        else:
            rank = 11
            inds = [0]
        ranks[index] = rank
        #print(inds[0])
        top1[index] = inds[0]
    #print(sum_corr / len(ranks))
    # print(ranks)
    r1 = len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    report_dict = dict()
    report_dict['r1'] = r1
    report_dict['r5'] = r5
    report_dict['r10'] = r10
    report_dict['r50'] = r50
    report_dict['medr'] = medr
    report_dict['meanr'] = meanr
    report_dict['sum'] = r1 + r5 + r50
    return report_dict, top1

def compute_retrieval_softneighbor(dot_product, len_dot_product):
    ranks = np.zeros(len_dot_product)
    top1 = np.zeros(len_dot_product)
    sn_margin = 5  #neighborhood margin
    for index in range(len_dot_product):
        inds = np.argsort(dot_product[index])[::-1]
        sn_inds = []
        for i_sn in range(-sn_margin, sn_margin + 1):
            idx_sn = min(len_dot_product - 1, max(0, (index + i_sn)))

            where = np.where(inds == idx_sn)
            #print(i_sn, idx_sn)
            #print(index, i_sn, idx_sn, where)
            sn_inds.append(where[0][0])
        rank = sn_inds[np.argsort(sn_inds)[0]]
        #print(sn_inds, rank)
        #print("=="*20)
        ranks[index] = rank
        top1[index] = inds[0]
    #print(sum(ranks < 0))
    r1 = len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    report_dict = dict()
    report_dict['r1'] = r1
    report_dict['r5'] = r5
    report_dict['r10'] = r10
    report_dict['r50'] = r50
    report_dict['medr'] = medr
    report_dict['meanr'] = meanr
    report_dict['sum'] = r1 + r5 + r50
    #print("R1 {}, R5 {}, R10 {}".format(r1, r5, r10))
    return report_dict, ranks

def compute_retrieval_cosine(dot_product, len_dot_product):
    ranks = np.zeros(len_dot_product)
    top1 = np.zeros(len_dot_product)
    ind_coarse_to_fine = []
    for index in range(len_dot_product):
        inds = np.argsort(dot_product[index])[::-1]
        inds_org = np.argmax(dot_product[index])
        where = np.where(inds == index)
        ind_coarse_to_fine.append(inds_org)
 
        rank = where[0][0]
        ranks[index] = rank
        top1[index] = inds[0]
    r1 = len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    report_dict = dict()
    report_dict['r1'] = r1
    report_dict['r5'] = r5
    report_dict['r10'] = r10
    report_dict['r50'] = r50
    report_dict['medr'] = medr
    report_dict['meanr'] = meanr
    report_dict['sum'] = r1 + r5 + r50
    return report_dict, top1, ind_coarse_to_fine


def compute_retrieval_metrics(dot_product):

    sort_similarity = np.sort(-dot_product, axis=1)
    diag_similarity = np.diag(-dot_product)
    diag_similarity = diag_similarity[:, np.newaxis]
    ranks = sort_similarity - diag_similarity
    ranks = np.where(ranks == 0)
    ranks = ranks[1]


    report_dict = dict()
    report_dict['r1'] = float(np.sum(ranks == 0)) / len(ranks)
    report_dict['r5'] = float(np.sum(ranks < 5)) / len(ranks)
    report_dict['r10'] = float(np.sum(ranks < 10)) / len(ranks)
    report_dict['r50'] = float(np.sum(ranks < 50)) / len(ranks)
    report_dict['medr'] = np.median(ranks) + 1
    report_dict['meanr'] = ranks.mean()
    report_dict[
        'sum'] = report_dict['r1'] + report_dict['r5'] + report_dict['r50']
    return report_dict, ranks


def get_logging_formatter():
    return logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                             datefmt="%m%d %H%M%S")


def get_timestamp_for_filename():
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    return ts


def get_logger_without_file(name, log_level="INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(get_logging_formatter())
    logger.addHandler(strm_hdlr)
    return logger


def get_logger(logdir, name, filename="run", log_level="INFO",
               log_file=True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = get_logging_formatter()
    if log_file:
        file_path = Path(logdir) / "{}_{}.log".format(
            filename,
            str(datetime.datetime.now()).split(".")[0].replace(
                " ", "_").replace(":", "_").replace("-", "_"))
        file_hdlr = logging.FileHandler(str(file_path))
        file_hdlr.setFormatter(formatter)
        logger.addHandler(file_hdlr)
    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)
    logger.addHandler(strm_hdlr)
    logger.propagate = False
    return logger


def close_logger(logger: logging.Logger):
    x = list(logger.handlers)
    for i in x:
        logger.removeHandler(i)
        i.flush()
        i.close()

# ---------- Profiling ----------

def profile_gpu_and_ram() -> Tuple[List[str], List[float], List[float], List[float], float, float, float]:
    """
    Profile GPU and RAM.

    Returns:
        GPU names, total / used memory per GPU, load per GPU, total / used / available RAM.
    """

    # get info from gputil
    _str, dct_ = _get_gputil_info()
    dev_num = os.getenv("CUDA_VISIBLE_DEVICES")
    if dev_num is not None:
        # single GPU set with OS flag
        gpu_info = [dct_[int(dev_num)]]
    else:
        # possibly multiple gpus, aggregate values
        gpu_info = []
        for dev_dict in dct_:
            gpu_info.append(dev_dict)

    # convert to GPU info and MB to GB
    gpu_names: List[str] = [gpu["name"] for gpu in gpu_info]
    total_memory_per: List[float] = [gpu["memoryTotal"] / 1024 for gpu in gpu_info]
    used_memory_per: List[float] = [gpu["memoryUsed"] / 1024 for gpu in gpu_info]
    load_per: List[float] = [gpu["load"] / 100 for gpu in gpu_info]

    # get RAM info and convert to GB
    mem = psutil.virtual_memory()
    ram_total: float = mem.total / 1024 ** 3
    ram_used: float = mem.used / 1024 ** 3
    ram_avail: float = mem.available / 1024 ** 3

    return gpu_names, total_memory_per, used_memory_per, load_per, ram_total, ram_used, ram_avail


def _get_gputil_info():
    """
    Returns info string for printing and list with gpu infos. Better formatting than the original GPUtil.

    Returns:
        gpu info string, List[Dict()] of values. dict example:
            ('id', 1),
            ('name', 'GeForce GTX TITAN X'),
            ('temperature', 41.0),
            ('load', 0.0),
            ('memoryUtil', 0.10645266950540452),
            ('memoryTotal', 12212.0)])]
    """

    gpus = GPUtil.getGPUs()
    attr_list = [
        {'attr': 'id', 'name': 'ID'}, {'attr': 'name', 'name': 'Name'},
        {'attr': 'temperature', 'name': 'Temp', 'suffix': 'C', 'transform': lambda x: x, 'precision': 0},
        {'attr': 'load', 'name': 'GPU util.', 'suffix': '% GPU', 'transform': lambda x: x * 100,
         'precision': 1},
        {'attr': 'memoryUtil', 'name': 'Memory util.', 'suffix': '% MEM', 'transform': lambda x: x * 100,
         'precision': 1}, {'attr': 'memoryTotal', 'name': 'Memory total', 'suffix': 'MB', 'precision': 0},
        {'attr': 'memoryUsed', 'name': 'Memory used', 'suffix': 'MB', 'precision': 0}
    ]
    gpu_strings = [''] * len(gpus)
    gpu_info = []
    for _ in range(len(gpus)):
        gpu_info.append({})

    for attrDict in attr_list:
        attr_precision = '.' + str(attrDict['precision']) if (
                'precision' in attrDict.keys()) else ''
        attr_suffix = str(attrDict['suffix']) if (
                'suffix' in attrDict.keys()) else ''
        attr_transform = attrDict['transform'] if (
                'transform' in attrDict.keys()) else lambda x: x
        for gpu in gpus:
            attr = getattr(gpu, attrDict['attr'])

            attr = attr_transform(attr)

            if isinstance(attr, float):
                attr_str = ('{0:' + attr_precision + 'f}').format(attr)
            elif isinstance(attr, int):
                attr_str = '{0:d}'.format(attr)
            elif isinstance(attr, str):
                attr_str = attr
            else:
                raise TypeError('Unhandled object type (' + str(
                    type(attr)) + ') for attribute \'' + attrDict[
                                    'name'] + '\'')

            attr_str += attr_suffix

        for gpuIdx, gpu in enumerate(gpus):
            attr_name = attrDict['attr']
            attr = getattr(gpu, attr_name)

            attr = attr_transform(attr)

            if isinstance(attr, float):
                attr_str = ('{0:' + attr_precision + 'f}').format(attr)
            elif isinstance(attr, int):
                attr_str = ('{0:' + 'd}').format(attr)
            elif isinstance(attr, str):
                attr_str = ('{0:' + 's}').format(attr)
            else:
                raise TypeError(
                    'Unhandled object type (' + str(
                        type(attr)) + ') for attribute \'' + attrDict[
                        'name'] + '\'')
            attr_str += attr_suffix
            gpu_info[gpuIdx][attr_name] = attr
            gpu_strings[gpuIdx] += '| ' + attr_str + ' '

    return "\n".join(gpu_strings), gpu_info
