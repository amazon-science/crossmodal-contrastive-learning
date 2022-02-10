import argparse
from copy import deepcopy
from collections import defaultdict
import datetime
import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Iterable, Union
import utils_collection.typext as typext
from pathlib import Path
import pathspec
import numpy as np

DEFAULT = "default"
REF = "ref"
NONE = "none"
LOGGER_NAME = "trainlog"
LOGGING_FORMATTER = logging.Formatter("%(levelname)5s %(message)s", datefmt="%m%d %H%M%S")


class CrossCLRLossConfig(typext.ConfigClass):
    """
    Contrastive loss Configuration Class

    Args:
        config: Configuration dictionary to be loaded, saving part.
    """

    def __init__(self, config: Dict) -> None:
        self.temperature: float = config.pop("temperature")
        self.temperature_weights: float = config.pop("temperature_weights")
        self.negative_weight: float = config.pop("negative_weight")
        self.score_thrshold: float = config.pop("score_thrshold")
        self.queue_size: float = config.pop("queue_size")

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    """
    Custom formatter
    - raw descriptions (no removing newlines)
    - show default values
    - show metavars (str, int, ...) instead of names
    - fit to console width
    """

    def __init__(self, prog: Any):
        try:
            term_size = os.get_terminal_size().columns
            max_help_pos = term_size // 2
        except OSError:
            term_size = None
            max_help_pos = 24
        super().__init__(
            prog, max_help_position=max_help_pos, width=term_size)

class TrainerPathConst(typext.ConstantHolder):
    """
    S
    tores directory and file names for training.
    """
    DIR_CONFIG = "config"
    DIR_EXPERIMENTS = "experiments"
    DIR_LOGS = "logs"
    DIR_MODELS = "models"
    DIR_METRICS = "metrics"
    DIR_EMBEDDINGS = "embeddings"
    DIR_TB = "tb"
    FILE_PREFIX_TRAINERSTATE = "trainerstate"
    FILE_PREFIX_MODEL = "model"
    FILE_PREFIX_OPTIMIZER = "optimizer"
    FILE_PREFIX_DATA = "data"
    FILE_PREFIX_METRICS_STEP = "metrics_step"
    FILE_PREFIX_METRICS_EPOCH = "metrics_epoch"

class LogLevelsConst(typext.ConstantHolder):
    """
    Loglevels, same as logging module.
    """
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0


class ArgParser(argparse.ArgumentParser):
    """
    ArgumentParser with Custom Formatter and some convenience functions.

    For best results, write a docstring at the top of the file and call it
    with ArgParser(description=__doc__)

    Args:
        description: Help text for Argparser. Set description=__doc__ and write help text into module header.
    """

    def __init__(self, description: str = "none"):
        super().__init__(description=description, formatter_class=CustomFormatter)


def match_folder(folder, exp_type: str, exp_group: str = None,
                 exp_list = None, search: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Match experiments in a folder.

    Args:
        folder: Folder of experiments to match, should be setup like FOLDER/EXP_TYPE/EXP_GROUP/EXP_NAME
        exp_type:
        exp_group:
        exp_list:
        search:

    Returns:
        Dictionary of experiment groups with a list of experiment names each.
    """
    logger = logging.getLogger(LOGGER_NAME)
    assert not (exp_list is not None and exp_group is not None), (
        "Cannot provide --exp_list and --exp_group at the same time.")

    # determine experiment group/name combinations to search
    exp_matcher_raw = []
    if exp_list is not None:
        # get experiment groups to search in from list
        exp_list_lines = Path(exp_list).read_text(encoding="utf8").splitlines(keepends=False)
        for line in exp_list_lines:
            line = line.strip()
            if line == "" or line[0] == "#":
                continue
            exp_matcher_raw.append(line)
    elif exp_group is not None:
        # get experiment groups from the argument
        for group in exp_group.split(","):
            exp_matcher_raw.append(group.strip())
    else:
        # include all groups and experiments
        exp_matcher_raw.append("*")
    matcher = create_string_matcher(exp_matcher_raw)

    # determine experiment name to search
    search_names = []
    if search is None:
        search_names.append("*")
    else:
        for name in search.split(","):
            search_names.append(name.strip())
    name_matcher = create_string_matcher(search_names)

    # determine root path and print infos
    root_path = Path(folder) / exp_type

    logger.info(f"Matching in {root_path} for --exp_group {exp_matcher_raw}, names --search {search_names}")

    # get all experiments and groups
    found = defaultdict(list)
    for new_exp_group in sorted(os.listdir(root_path)):
        for new_exp_name in sorted(os.listdir(root_path / new_exp_group)):
            # when searching configs, remove the .yaml ending
            if new_exp_name.endswith(".yaml"):
                new_exp_name = new_exp_name[:-5]
            # match group and name
            match_str = f"{new_exp_group}/{new_exp_name}"
            if matcher.match_file(match_str) and name_matcher.match_file(new_exp_name):
                found[new_exp_group].append(new_exp_name)

    logger.debug(f"Found: {found}")

    return found


def create_string_matcher(pattern: Union[str, List[str]]) -> pathspec.PathSpec:
    """
    Given one or several patterns with the syntax of a .gitignore file, create a matcher object that can
    be used to match strings against the pattern.

    Args:
        pattern: One or several patterns.

    Returns:
        PathSpec matcher object.
    """
    if isinstance(pattern, str):
        pattern = [pattern]
    matcher = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, pattern)
    return matcher

def create_logger_without_file(name: str, log_level: int = LogLevelsConst.INFO, no_parent: bool = False,
                               no_print: bool = False) -> logging.Logger:
    """
    Create a stdout only logger.

    Args:
        name: Name of the logger.
        log_level: Verbosity level.
        no_parent: Disable parents, can be used to avoid duplicate log entries.
        no_print: Do not print a message on creation.
    Returns:
        Created logger.
    """
    return create_logger(name, log_dir="", log_level=log_level, no_parent=no_parent, no_print=no_print)


def create_logger(
        name: str, *, filename: str = "run", log_dir= "", log_level: int = LogLevelsConst.INFO,
        no_parent: bool = False, no_print: bool = False) -> logging.Logger:
    """
    Create a new logger.

    Notes:
        This created stdlib logger can later be retrieved with logging.getLogger(name) with the same name.
        There is no need to pass the logger instance between objects.

    Args:
        name: Name of the logger.
        log_dir: Target logging directory. Empty string will not create files.
        filename: Target filename.
        log_level: Verbosity level.
        no_parent: Disable parents, can be used to avoid duplicate log entries.
        no_print: Do not print a message on creation.

    Returns:
    """
    # create logger, set level
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # remove old handlers to avoid duplicate messages
    remove_handlers_from_logger(logger)

    # file handler
    file_path = None
    if log_dir != "":
        ts = get_timestamp_for_filename()
        file_path = Path(log_dir) / "{}_{}.log".format(filename, ts)
        file_hdlr = logging.FileHandler(str(file_path))
        file_hdlr.setFormatter(LOGGING_FORMATTER)
        logger.addHandler(file_hdlr)

    # stdout handler
    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(LOGGING_FORMATTER)
    logger.addHandler(strm_hdlr)

    # disable propagating to parent to avoid double logs
    if no_parent:
        logger.propagate = False

    if not no_print:
        print(f"Logger: '{name}' to {file_path}")
    return logger

def np_str_len(str_arr) -> np.ndarray:
    """
    Fast way to get string length in a numpy array with datatype string.

    Args:
        str_arr: Numpy array of strings with arbitrary shape.

    Returns:
        Numpy array of string lengths, same shape as input.

    Notes:
        Source: https://stackoverflow.com/questions/44587746/length-of-each-string-in-a-numpy-array
        The latest improved answers don't really work. This code should work for all except strange special characters.
    """
    if not isinstance(str_arr, np.ndarray):
        # also support iterables of strings
        str_arr = np.array(str_arr)
    # check input type
    if str(str_arr.dtype)[:2] != "<U":
        raise TypeError(
            f"Computing string length of dtype {str_arr.dtype} will not work correctly. Cast array to string first.")

    # see the link in the docstring as an explanation of what exactly is happening here
    try:
        v = str_arr.view(np.uint32).reshape(str_arr.size, -1)
    except TypeError as e:
        print(f"Input {str_arr} shape {str_arr.shape} dtype {str_arr.dtype}")
        raise e
    len_arr = np.argmin(v, 1)
    len_arr[v[np.arange(len(v)), len_arr] > 0] = v.shape[-1]
    len_arr = np.reshape(len_arr, str_arr.shape)
    return len_arr

def remove_handlers_from_logger(logger: logging.Logger) -> None:
    """
    Remove handlers from the logger.

    Args:
        logger: Logger.
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.flush()
        handler.close()


def print_logger_info(logger: logging.Logger) -> None:
    """
    Print infos describing the logger: The name and handlers.

    Args:
        logger: Logger.
    """
    print(logger.name)
    x = list(logger.handlers)
    for i in x:
        handler_str = f"Handler {i.name} Type {type(i)}"
        print(handler_str)

class ExperimentTypesConst(typext.ConstantHolder):
    """
    Store model types for COOT.
    """
    CAPTIONING = "captioning"
    RETRIEVAL = "retrieval"

class MetersConst(typext.ConstantHolder):
    """
    Additional metric fields.
    """
    TRAIN_LOSS_CC = "train/loss_cc"
    TRAIN_LOSS_CONTRASTIVE = "train/loss_contr"
    VAL_LOSS_CC = "val/loss_cc"
    VAL_LOSS_CONTRASTIVE = "val/loss_contr"
    TEST_LOSS_CONTRASTIVE = "test/loss_contr"
    RET_MODALITIES = ["vid2par", "par2vid", "clip2sent", "sent2clip"]
    RET_MODALITIES_SHORT = ["v2p", "p2v", "c2s", "s2c"]
    RET_METRICS = ["r1", "r5", "r10", "r50", "medr", "meanr"]

    TRAIN_EPOCH = "train_base/epoch"
    TIME_TOTAL = "ztime/time_total"
    TIME_VAL = "ztime/time_val"
    VAL_LOSS = "val_base/loss"
    TEST_LOSS = "test_base/loss"

    VAL_BEST_FIELD = "val_base/best_field"
    TRAIN_LR = "train_base/lr"
    PROFILE_GPU_MEM_PERCENT = "zgpu/mem_percent"
    PROFILE_GPU_MEM_USED = "zgpu/mem_used"
    TIME_STEP_FORWARD = "ztime/step_forward"
    TIME_STEP_BACKWARD = "ztime/step_backward"
    TIME_STEP_TOTAL = "ztime/step_total"
    TIME_STEP_OTHER = "ztime/step_other"
    TRAIN_GRAD_CLIP = "train_base/grad_clip_total_norm"
    TRAIN_LOSS = "train_base/loss"
    PROFILE_GPU_LOAD = "zgpu/load"
    # not logged
    PROFILE_GPU_MEM_TOTAL = "zgpu/mem_total"
    PROFILE_RAM_TOTAL = "zram/total"
    PROFILE_RAM_USED = "zram/used"
    PROFILE_RAM_AVAILABLE = "zram/avail"
class ConfigNamesConst(typext.ConstantHolder):
    """
    Stores configuration group names.
    """
    TRAIN = "training"
    VAL = "val"
    TEST = "test"
    DATASET = "dataset"
    LOGGING = "logging"
    SAVING = "saving"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "scheduler"

class RetrievalNetworksConst(typext.ConstantHolder):
    """
    Store network names for COOT.
    """
    NET_VIDEO_LOCAL = "video_local"
    NET_VIDEO_GLOBAL = "video_global"
    NET_TEXT_LOCAL = "text_local"
    NET_TEXT_GLOBAL = "text_global"

class ConfigClass:
    """
    Base class for config storage classes. Defines representation for printing.
    """

    def __repr__(self) -> str:
        """
        Represent class attributes as key, value pairs.

        Returns:
            String representation of the config.
        """
        str_repr = ["", "-" * 10 + " " + type(self).__name__]
        for key, value in vars(self).items():
            if key in ["config_orig"]:
                continue
            if isinstance(value, ConfigClass):
                # str_repr += ["-" * 10 + " " + key, str(value)]
                str_repr += [str(value)]
            else:
                str_repr += [f"    {key} = {value}"]
        return "\n".join(str_repr)

class LossesConst():
    CONTRASTIVE = "contrastive"
    CROSSENTROPY = "crossentropy"
class ContrastiveLossConfig(ConfigClass):
    """
    Contrastive loss Configuration Class

    Args:
        config: Configuration dictionary to be loaded, saving part.
    """

    def __init__(self, config: Dict) -> None:
        self.margin: float = config.pop("margin")
        self.weight_high: float = config.pop("weight_high")
        self.weight_high_cluster: float = config.pop("weight_high_cluster")
        self.weight_low: float = config.pop("weight_low")
        self.weight_low_cluster: float = config.pop("weight_low_cluster")
        self.weight_context: float = config.pop("weight_context")
        self.weight_context_cluster: float = config.pop("weight_context_cluster")

class BaseLoggingConfig(ConfigClass):
    """
    Base Logging Configuration Class

    Args:
        config: Configuration dictionary to be loaded, logging part.
    """

    def __init__(self, config: Dict) -> None:
        self.step_train: int = config.pop("step_train")
        self.step_val: int = config.pop("step_val")
        self.step_test: int = config.pop("step_test")
        self.step_gpu: int = config.pop("step_gpu")
        self.step_gpu_once: int = config.pop("step_gpu_once")
        assert self.step_train >= -1
        assert self.step_val >= -1
        assert self.step_gpu >= -1
        assert self.step_gpu_once >= -1

class BaseSavingConfig(ConfigClass):
    """
    Base Saving Configuration Class

    Args:
        config: Configuration dictionary to be loaded, saving part.
    """

    def __init__(self, config: Dict) -> None:
        self.keep_freq: int = config.pop("keep_freq")
        self.save_last: bool = config.pop("save_last")
        self.save_best: bool = config.pop("save_best")
        self.save_opt_state: bool = config.pop("save_opt_state")
        assert self.keep_freq >= -1


class BaseTrainerState(typext.SaveableBaseModel):
    """
    Current trainer state that must be saved for training continuation..
    """
    # total time bookkeeping
    time_total: float = 0
    time_val: float = 0
    # state info TO SAVE
    start_epoch: int = 0
    current_epoch: int = 0
    epoch_step: int = 0
    total_step: int = 0
    det_best_field_current: float = 0
    det_best_field_best: Optional[float] = None

    # state info lists
    infos_val_epochs: List[int] = []
    infos_val_steps: List[int] = []
    infos_val_is_good: List[int] = []

    # logging
    last_grad_norm: int = 0
    
class BaseTrainConfig(ConfigClass):
    """
    Base configuration class for training.

    Args:
        config: Configuration dictionary to be loaded, training part.
    """

    def __init__(self, config: Dict) -> None:
        self.batch_size: int = config.pop("batch_size")
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        self.num_epochs: int = config.pop("num_epochs")
        assert isinstance(self.num_epochs, int) and self.num_epochs > 0
        self.loss_func: str = config.pop("loss_func")
        assert isinstance(self.loss_func, str)
        self.debug_size: int = config.pop("debug_size")
        if self.loss_func == LossesConst.CONTRASTIVE:
            self.contrastive_loss_config = ContrastiveLossConfig(config.pop("contrastive_loss_config"))
            self.cross_clr_config = CrossCLRLossConfig(config.pop("cross_clr_config"))


class BaseValConfig(ConfigClass):
    """
    Base configuration class for validation.

    Args:
        config: Configuration dictionary to be loaded, validation part.
    """

    def __init__(self, config: Dict) -> None:
        self.batch_size: int = config.pop("batch_size")
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        self.val_freq: int = config.pop("val_freq")
        assert isinstance(self.val_freq, int) and self.val_freq > 0
        self.val_start: int = config.pop("val_start")
        assert isinstance(self.val_start, int) and self.val_start >= 0
        self.debug_size: int = config.pop("debug_size")

        # self.val_train_set: bool = config.pop("val_train_set")
        # assert isinstance(self.val_train_set, bool)
        self.det_best_field: str = config.pop("det_best_field")
        assert isinstance(self.det_best_field, str)
        self.det_best_compare_mode: str = config.pop("det_best_compare_mode")
        assert isinstance(self.det_best_compare_mode, str) and self.det_best_compare_mode in ["min", "max"]
        self.det_best_threshold_mode: str = config.pop("det_best_threshold_mode")
        assert isinstance(self.det_best_threshold_mode, str) and self.det_best_threshold_mode in ["rel", "abs"]
        self.det_best_threshold_value: float = config.pop("det_best_threshold_value")
        assert isinstance(self.det_best_threshold_value, (int, float)) and self.det_best_threshold_value >= 0
        self.det_best_terminate_after: float = config.pop("det_best_terminate_after")
        assert isinstance(self.det_best_terminate_after, int) and self.det_best_terminate_after >= -1

class BaseTestConfig(ConfigClass):
    """
    Base configuration class for Test.

    Args:
        config: Configuration dictionary to be loaded, test part.
    """

    def __init__(self, config: Dict) -> None:
        self.batch_size: int = config.pop("batch_size")
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        self.test_freq: int = config.pop("test_freq")
        self.test_start: int = config.pop("test_start")
        self.debug_size: int = config.pop("debug_size")




class BaseSavingConfig(ConfigClass):
    """
    Base Saving Configuration Class

    Args:
        config: Configuration dictionary to be loaded, saving part.
    """

    def __init__(self, config: Dict) -> None:
        self.keep_freq: int = config.pop("keep_freq")
        self.save_last: bool = config.pop("save_last")
        self.save_best: bool = config.pop("save_best")
        self.save_opt_state: bool = config.pop("save_opt_state")
        assert self.keep_freq >= -1

class BaseExperimentConfig(ConfigClass):
    """
    Base configuration class, loads the dict from yaml config files for an experiment.

    This is where the entire config dict will be loaded into first.

    Args:
        config: Configuration dictionary to be loaded.
    """

    def __init__(self, config: Dict, strict: bool = True) -> None:
        self.config_orig = deepcopy(config)  # backup original input dict
        self.config = config  # bind dict to class
        self.strict = strict
        resolve_sameas_config_recursively(config)  # resolve "same_as" reference fields to dictionary objects.
        self.description: str = config.pop("description", "no description given.")
        self.random_seed: Optional[int] = config.pop("random_seed")
        self.config_type: str = config.pop("config_type")
        self.use_cuda: bool = config.pop("use_cuda")
        self.use_multi_gpu: bool = config.pop("use_multi_gpu")
        self.cudnn_enabled: bool = config.pop("cudnn_enabled")
        self.cudnn_benchmark: bool = config.pop("cudnn_benchmark")
        self.cudnn_deterministic: bool = config.pop("cudnn_deterministic")
        self.cuda_non_blocking: bool = config.pop("cuda_non_blocking")
        self.fp16_train: bool = config.pop("fp16_train")
        self.fp16_val: bool = config.pop("fp16_val")

    def post_init(self):
        """
        Check config dict for correctness and raise

        Returns:
        """
        if self.strict:
            check_config_dict(self.__class__.__name__, self.config)
class MetricComparisonConst(typext.ConstantHolder):
    """
    Fields for the early stopper.
    """
    # metric comparison
    VAL_DET_BEST_MODE_MIN = "min"
    VAL_DET_BEST_MODE_MAX = "max"
    VAL_DET_BEST_TH_MODE_REL = "rel"
    VAL_DET_BEST_TH_MODE_ABS = "abs"

class RetrievalConfig(BaseExperimentConfig):
    """
    Definition to load the yaml config files for training a retrieval model. This is where the actual config dict
    goes and is processed.

    Args:
        config: Configuration dictionary to be loaded, logging part.
        is_train: Whether there will be training or not.
    """

    def __init__(self, config: Dict[str, Any], *, is_train: bool = True) -> None:
        super().__init__(config)
        self.name = "config_ret"

        if not is_train:
            # Disable dataset caching
            logger = logging.getLogger(LOGGER_NAME)
            logger.debug("Disable dataset caching during validation.")
            config["dataset_val"]["preload_vid_feat"] = False
            config["dataset_val"]["preload_text_feat"] = False

        try:
            self.train = RetrievalTrainConfig(config.pop(ConfigNamesConst.TRAIN))
            self.val = RetrievalValConfig(config.pop(ConfigNamesConst.VAL))
            self.test = RetrievalTestConfig(config.pop(ConfigNamesConst.TEST))
            self.dataset = RetrievalDatasetConfig(config.pop(ConfigNamesConst.DATASET))
            self.logging = BaseLoggingConfig(config.pop(ConfigNamesConst.LOGGING))
            self.saving = BaseSavingConfig(config.pop(ConfigNamesConst.SAVING))
            self.optimizer = OptimizerConfig(config.pop(ConfigNamesConst.OPTIMIZER))
            self.lr_scheduler = SchedulerConfig(config.pop(ConfigNamesConst.LR_SCHEDULER))

            self.model_cfgs = {}
            for key in RetrievalNetworksConst.values():
                self.model_cfgs[key] = TransformerConfig(config.pop(key))
        except KeyError as e:
            print()
            print(traceback.format_exc())
            print(f"ERROR: {e} not defined in config {self.__class__.__name__}\n")
            raise e

        self.post_init()


class RetrievalValConfig(BaseValConfig):
    """
    Retrieval validation configuration class.

    Args:
        config: Configuration dictionary to be loaded, training part.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.val_clips: bool = config.pop("val_clips")
        assert isinstance(self.val_clips, bool)
        self.val_clips_freq: int = config.pop("val_clips_freq")
        assert isinstance(self.val_clips_freq, int)

class RetrievalTestConfig(BaseTestConfig):
    """
    Retrieval test configuration class.

    Args:
        config: Configuration dictionary to be loaded, training part.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.test_clips: bool = config.pop("test_clips")
        assert isinstance(self.test_clips, bool)
        self.test_clips_freq: int = config.pop("test_clips_freq")
        assert isinstance(self.test_clips_freq, int)

class RetrievalTrainConfig(BaseTrainConfig):
    """
    Retrieval trainer configuration class.

    Args:
        config: Configuration dictionary to be loaded, training part.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)


class BaseDatasetConfig(ConfigClass):
    """
    Base Dataset Configuration class

    Args:
        config: Configuration dictionary to be loaded, dataset part.
    """

    def __init__(self, config: Dict) -> None:
        # general dataset info
        self.name: str = config.pop("name")
        # self.data_type: str = config.pop("data_type")
        # self.subset: str = config.pop("subset")
        self.train_split: str = config.pop("train_split")
        self.val_split: str = config.pop("val_split")
        self.test_split: str = config.pop("test_split")
        # general dataloader configuration
        self.pin_memory: bool = config.pop("pin_memory")
        self.num_workers: int = config.pop("num_workers")
        self.drop_last: bool = config.pop("drop_last")

class RetrievalDatasetConfig(BaseDatasetConfig):
    """
    Retrieval dataset configuration class.

    Args:
        config: Configuration dictionary to be loaded, dataset part.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        # self.metadata_name: str = config.pop("metadata_name")
        feat_tmp: str = config.pop("modality_feat_name_a")
        v_name_list = feat_tmp.split(",")
        self.modality_feat_name_a = []
        self.modality_feat_dim = {}
        for v_name in v_name_list:
            self.modality_feat_name_a.append(v_name)
            self.modality_feat_dim[v_name]: int = config.pop("{}_feat_dim".format(v_name))

        self.modality_feat_dim["text"]: int = config.pop("text_feat_dim")
        self.modality_feat_name_b: str = config.pop("modality_feat_name_b")
        # self.text_feat_dim: int = config.pop("text_feat_dim")
        self.min_frames: int = config.pop("min_frames")  # unused
        self.max_frames: int = config.pop("max_frames")
        self.feat_agg_dim: int = config.pop("feat_agg_dim")
        self.use_clips: bool = config.pop("use_clips")  # unused
        self.add_stop_frame: int = config.pop("add_stop_frame")
        self.expand_segments: int = config.pop("expand_segments")
        self.preload_data: bool = config.pop("preload_data")
        self.frames_noise: int = config.pop("frames_noise")

        assert isinstance(self.modality_feat_name_a, list)
        assert isinstance(self.modality_feat_name_b, str)
        assert isinstance(self.min_frames, int)
        assert isinstance(self.max_frames, int)
        assert isinstance(self.modality_feat_dim, dict)
        # assert isinstance(self.text_feat_dim, int)
        assert isinstance(self.use_clips, bool)
        assert isinstance(self.add_stop_frame, int)
        assert isinstance(self.expand_segments, int)
        assert isinstance(self.preload_data, bool)

class OptimizerConst(typext.ConstantHolder):
    """
    Optimizer name constants.
    """
    ADAM = "adam"
    RADAM = "radam"


class OptimizerConfig(ConfigClass):
    """
    Optimizer Configuration Class

    Args:
        config: Configuration dictionary to be loaded, optimizer part.
    """

    def __init__(self, config: Dict) -> None:
        self.name: str = config.pop("name")
        self.lr: float = config.pop("lr")
        self.weight_decay: float = config.pop("weight_decay")
        self.weight_decay_for_bias: bool = config.pop("weight_decay_for_bias")
        self.momentum: float = config.pop("momentum")
        self.adam_beta2: float = config.pop("adam_beta2")
        self.adam_eps: float = config.pop("adam_eps")
        self.lr_decay_mult: bool = config.pop("lr_decay_mult")

class SchedulerConfig(ConfigClass):
    """
    Scheduler Configuration Class

    Args:
        config: Configuration dictionary to be loaded, scheduler part.
    """

    def __init__(self, config: Dict) -> None:
        # scheduler name
        # warmup can be enabled for all schedulers
        self.warmup_epochs: int = config.pop("warmup")

        # fields required for reduce on plateau scheduler
        self.rop_patience: int = config.pop("patience")
        self.rop_cooldown: int = config.pop("cooldown")

class TransformerEncoderConfig(ConfigClass):
    """
    TransformerEncoder Submodule

    Args:
        config: Configuration dictionary to be loaded.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        # load fields required for a transformer
        self.hidden_dim: int = config.pop("hidden_dim")
        self.num_layers: int = config.pop("num_layers")
        self.dropout: float = config.pop("dropout")
        self.num_heads: int = config.pop("num_heads")
        self.pointwise_ff_dim: int = config.pop("pointwise_ff_dim")
        self.activation = ActivationConfig(config.pop("activation"))
        self.norm = NormalizationConfig(config.pop("norm"))

class TransformerConfig(ConfigClass):
    """
    Configuration class for a single coot network

    Args:
        config: Configuration dictionary to be loaded, dataset part.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.name: str = config.pop("name")
        self.output_dim: int = config.pop("output_dim")  # output dim must be specified for future modules in the chain
        # self.dropout: float = config.pop("dropout")

        # Add input FC to downsample input features to the transformer dimension
        self.use_input_fc: bool = config.pop("input_fc")
        # if self.use_input_fc:
        self.input_fc_dim = config.pop("input_fc_output_dim")

        # Self-attention
        self.selfatn = TransformerEncoderConfig(config.pop("selfatn_config"))


        # cross-attention
        self.use_context: bool = config.pop("use_context")

        self.use_subspace: bool = config.pop("use_subspace")

        assert isinstance(self.use_context, bool)
        # if self.use_context:
        #     # fields required for cross-attention
        #     self.crossatn = TransformerEncoderConfig(config.pop("selfatn_config"))
        # pooler
        self.pooler_config = PoolerConfig(config.pop("pooler_config"))

        # weight initialiazion
        self.weight_init_type: str = config.pop("weight_init_type")
        self.weight_init_std: float = config.pop("weight_init_std")


# ---------- Time utilities ----------

def get_timestamp_for_filename(dtime = None):
    """
    Convert datetime to timestamp for filenames.

    Args:
        dtime: Optional datetime object, will use now() if not given.

    Returns:
    """
    if dtime is None:
        dtime = datetime.datetime.now()
    ts = str(dtime).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    return ts


# ---------- Files ----------
def parse_file_to_list(file) -> List[str]:
    """
    Given a text file, read contents to list of lines. Strip lines, ignore empty and comment lines

    Args:
        file: Input file.

    Returns:
        List of lines.
    """
    # loop lines
    output = []
    for line in Path(file).read_text(encoding="utf8").splitlines(keepends=False):
        # strip line
        line = line.strip()
        if line == "":
            # skip empty line
            continue
        if line[0] == "#":
            # skip comment line
            continue
        # collect
        output.append(line)
    return output


# ---------- Config / dict ----------

def resolve_sameas_config_recursively(config: Dict, *, root_config: Optional[Dict] = None):
    """
    Recursively resolve config fields described with same_as.

    If any container in the config has the field "same_as" set, find the source identifier and copy all data
    from there to the target container. The source identifier can nest with dots e.g.
    same_as: "net_video_local.input_fc_config" will copy the values from container input_fc_config located inside
    the net_video_local container.

    Args:
        config: Config to modify.
        root_config: Config to get the values from, usually the same as config.

    Returns:
    """
    if root_config is None:
        root_config = config
    # loop the current config and check
    loop_keys = list(config.keys())
    for key in loop_keys:
        value = config[key]
        if not isinstance(value, dict):
            continue
        same_as = value.get("same_as")
        if same_as is not None:
            # current container should be filled with the values from the source container. loop source container
            source_container = get_dict_value_recursively(root_config, same_as)
            for key_source, val_source in source_container.items():
                # only write fields that don't exist yet, don't overwrite everything
                if key_source not in config[key]:
                    # at this point we want a deepcopy to make sure everything is it's own object
                    config[key][key_source] = deepcopy(val_source)
            # at this point, remove the same_as field.
            del value["same_as"]

        # check recursively
        resolve_sameas_config_recursively(config[key], root_config=root_config)


def get_dict_value_recursively(dct: Dict, key: str) -> Any:
    """
    Nest into the dict given a key like root.container.subcontainer

    Args:
        dct: Dict to get the value from.
        key: Key that can describe several nesting steps at one.

    Returns:
        Value.
    """
    key_parts = key.split(".")
    if len(key_parts) == 1:
        # we arrived at the leaf of the dict tree and can return the value
        return dct[key_parts[0]]
    # nest one level deeper
    return get_dict_value_recursively(dct[key_parts[0]], ".".join(key_parts[1:]))

def check_config_dict(name: str, config: Dict[str, Any], strict: bool = True) -> None:
    """
    Make sure config has been read correctly with .pop(), and no fields are left over.

    Args:
        name: config name
        config: config dict
        strict: Throw errors
    """
    remaining_keys, remaining_values = [], []
    for key, value in config.items():
        if key == REF:
            # ignore the reference configurations, they can later be used for copying things with same_as
            continue
        remaining_keys.append(key)
        remaining_values.append(value)
    # check if something is left over
    if len(remaining_keys) > 0:
        if not all(value is None for value in remaining_values):
            err_msg = (
                f"keys and values remaining in config {name}: {remaining_keys}, {remaining_values}. "
                f"Possible sources of this error: Typo in the field name in the yaml config file. "
                f"Incorrect fields given with --config flag. "
                f"Field should be added to the config class so it can be parsed. "
                f"Using 'same_as' and forgot to set these fields to null.")

            if strict:
                print(f"Print config for debugging: {config}")
                raise ValueError(err_msg)
            logging.getLogger(LOGGER_NAME).warning(err_msg)


class NormalizationConst(typext.ConstantHolder):
    """
    Define normalization module names.
    """
    LAYERNORM_PYTORCH = "layernorm_pytorch"
    LAYERNORM_COOT = "layernorm_coot"


class NormalizationConfig(ConfigClass):
    """
    Normalization config object. Stores hyperparameters.

    Examples:
        >>> NormalizationConfig("layernorm")
        >>> NormalizationConfig({"name": "layernorm", "affine": "false"})

    Args:
        name_or_config: Either provide string name of the activation function (e.g. "layernorm") or a dict with name and
            hyperparameters (e.g. {"name": "layernorm", "epsilon": 1e-6})
    """

    def __init__(self, name_or_config):
        # Determine if configuration is given by a string name or a config dict.
        if isinstance(name_or_config, str):
            config: Dict[str, Any] = {}
            self.name = name_or_config
        elif isinstance(name_or_config, dict):
            config = name_or_config
            self.name = config.pop("name")
        else:
            raise ValueError(f"Type {name_or_config} not understood.")
        # Set optional fields
        self.eps: float = config.pop("eps", 1e-6)
        self.affine: bool = config.pop("affine", True)
        # StochNorm
        self.momentum: float = config.pop("momentum", 0.1)
        self.track_running_stats = config.pop("track_running_stats", True)

class ActivationConst(typext.ConstantHolder):
    RELU = "relu"
    GELU = "gelu"
    LEAKYRELU = "leakyrelu"  # params: negative_slope (default 1/100)


class ActivationConfig(ConfigClass):
    """
    Activation function.

    Examples:
        >>> ActivationConfig("relu")
        >>> ActivationConfig({"name": "leakyrelu", "negative_slope": 1e-2})

    Args:
        name_or_config: Either provides string name of the activation or a dict with name and hyperparameters.
    """

    def __init__(self, name_or_config):
        # Determine if configuration is given by a string name or a config dict.
        if isinstance(name_or_config, str):
            config = {}
            self.name = name_or_config
        else:
            config = name_or_config
            self.name = config.pop("name")
        # Set optional fields
        self.negative_slope = config.pop("negative_slope", 1e-2)

class PoolerConst(typext.ConstantHolder):
    """
    Pooler types for coot.

    Notes:
        ATN: Attention-aware feature aggregation
        AVG_SPECIAL: Average-pooling as described in the appendix
    """
    ATN = "atn"
    AVG_SPECIAL = "avg_special"


class PoolerConfig(ConfigClass):
    """
    Pooling Submodule

    Args:
        name_or_config: Either provide string name of or a dict with name and hyperparameters.
    """

    def __init__(self, name_or_config):
        # Determine if configuration is given by a string name or a config dict.
        if isinstance(name_or_config, str):
            config: Dict[str, Any] = {}
            self.name = name_or_config
        elif isinstance(name_or_config, dict):
            config = name_or_config
            self.name = config.pop("name")
        else:
            raise ValueError(f"Type {name_or_config} not understood.")
        # fields required for attention-based pooling
        self.hidden_dim: int = config.pop("hidden_dim", 0)
        self.num_heads: int = config.pop("num_heads", 1)
        self.num_layers: int = config.pop("num_layers", 1)
        self.dropout: float = config.pop("dropout", 0)
        self.activation = ActivationConfig(config.pop("activation", "relu"))