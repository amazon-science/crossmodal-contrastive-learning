import json
from collections import OrderedDict
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
import torch.utils.data as data
from easydict import EasyDict
from tqdm import tqdm

from utils_collection import general
from utils_collection.text_embedding import preprocess_bert_paragraph

import pickle as pck
import os 

DBG_LOADER_LIMIT = 10000

class BertTextFeatureLoader:
    def __init__(
            self, dataset_path_dict: EasyDict, ids, preload=True, debug_size: int=DBG_LOADER_LIMIT):
        self.h5_path = dataset_path_dict["language_feats"]
        lens_file = dataset_path_dict["meta_text_len"]
        l_file = lens_file.open("rt", encoding="utf8")
        self.lens = json.load(l_file)
        l_file.close()
        self.cached_data = None

        if preload:
            h5file = h5py.File(self.h5_path, "r")
            self.cached_data = {}
            mod_keys = list(ids.keys())
            i = 0
            for id_ in tqdm(ids[mod_keys[0]], desc="preload text"):
                np_array = h5file[id_]
                shared_array = general.make_shared_array(np_array)
                self.cached_data[id_] = shared_array
                if debug_size < DBG_LOADER_LIMIT: # For quick debugging we limit the data loading to debug_size
                    if i > debug_size:
                        break
                    i += 1
            h5file.close()

    def __getitem__(self, id_):
        lens = self.lens[id_]
        if self.cached_data is None:
            h5file = h5py.File(self.h5_path, "r")
            features = np.array(h5file[id_])
            h5file.close()
            return features, lens
        return self.cached_data[id_], lens


class ActivityNetVideoFeatureLoader:
    def __init__(self, dataset_path: Path, ids: List[str], preload: bool):
        self.dataset_path = Path(dataset_path)
        self.features_path = (dataset_path / "features" /
                              "ICEP_V3_global_pool_skip_8_direct_resize")
        self.cached_data = None
        if preload:
            self.cached_data = {}
            for id_ in tqdm(ids, desc="preload videos"):
                np_array = self.load_from_file(id_)
                shared_array = general.make_shared_array(np_array)
                self.cached_data[id_] = shared_array

    def __getitem__(self, id_):
        if self.cached_data is None:
            return self.load_from_file(id_)
        else:
            return self.cached_data[id_]

    def load_from_file(self, id_):
        return np.load(str(self.features_path / f"{id_}.npz"))[
            "frame_scores"].squeeze(1).squeeze(2).squeeze(2)

class LSMDCVideoFeatureLoader:
    def __init__(
            self, dataset_path_dict: EasyDict, ids: List[str],
            preload: bool, data_split, debug_size: int=DBG_LOADER_LIMIT):
        self.h5_path = {}
        for i_mod in dataset_path_dict["video_feats"]:
            self.h5_path[i_mod] = dataset_path_dict["video_feats"][i_mod]
        self.cached_data = None
        self.features_source = "h5"
        self.data_keys = ids
        self.data_split = data_split
        if preload:
            self.cached_data = {}
            # buffer data to memory
            for i_mod in dataset_path_dict["video_feats"]:
                self.cached_data[i_mod] = {}
            for i_mod in dataset_path_dict["video_feats"]:
                h5 = h5py.File(self.h5_path[i_mod], "r")
                i = 0
                for key in tqdm(self.data_keys[i_mod], desc="Preloading {} - modality: {} ==>".format(self.data_split, i_mod)):
                    data = h5[key]
                    self.cached_data[i_mod][key] = general.make_shared_array(data)
                    if debug_size < DBG_LOADER_LIMIT: # limit the dataloading for quick debugging
                        if i > debug_size:
                            break
                        i += 1
    def get_features_by_key(self, item: str) -> np.ndarray:
        """
        Given feature key, load the feature.

        Args:
            item: Key.

        Returns:
            Feature data array with shape (num_frames, feature_dim)
        """
        if self.features_source == "h5":
            # load from h5
            h5 = h5py.File(self.h5_path, "r")
            return np.array(h5[item])
        if self.features_source == "npz_activitynet":
            # load from single npz file
            # this is specific to the activitynet inception features
            return np.load(str(self.dataset_path / "features" / self.features_name / f"v_{item}.npz")
                           )["frame_scores"].squeeze(1).squeeze(2).squeeze(2)
        raise NotImplementedError(f"Feature source type {self.features_source} not understood.")

    def get_features_as_items(self, load_all: bool = False, i_mod: str = "action"):
        """
        Iterator for key, value pairs of all features.

        Args:
            load_all: If true, ignores the provided data keys and loops everything in the path.

        Yields:
            Tuple of feature key and feature data array with shape (num_frames, feature_dim)
        """
        if self.features_source == "h5":
            # load from h5
            h5 = h5py.File(self.h5_path[i_mod], "r")
            
            if load_all:
                for key, data in h5.items():
                    yield key, data
            else:
                for key in self.data_keys:
                    yield key, h5[key]
        elif self.features_source == "npz_activitynet":
            # load from npz for activitynet
            if load_all:
                files = os.listdir(self.dataset_path / "features" / self.features_name)
                for file in files:
                    data_key = file[2:-4]  # extract youtube id from v_###########.npz
                    yield data_key, self.get_features_by_key(data_key)
            else:
                for data_key in self.data_keys:
                    yield data_key, self.get_features_by_key(data_key)
        else:
            raise NotImplementedError(f"Feature source type {self.features_source} not understood.")

    def __getitem__(self, id_):
        if self.cached_data is None:
            h5file = h5py.File(self.h5_path, "r")
            features = np.array(h5file[id_])
            h5file.close()
            return features
        else:
            return self.cached_data[id_]

class LSMDCVideoPickleLoader:
    def __init__(
            self, dataset_path_dict: EasyDict, ids: List[str],
            preload: bool, data_split, debug_size: int=DBG_LOADER_LIMIT):
        self.h5_path = {}
        for i_mod in dataset_path_dict["video_feats"]:
            self.h5_path[i_mod] = dataset_path_dict["video_feats"][i_mod]
        self.cached_data = None
        self.features_source = "h5"
        self.data_keys = ids
        self.data_split = data_split
        self.pickle_folder = dataset_path_dict["pickle_path"]
        self.pck_folder = os.path.join(self.pickle_folder, self.data_split)

        self.cached_data = None
        i = 0
        if preload:
            self.cached_data = {}
            mod_name = list(dataset_path_dict["video_feats"])[0]
            for id_ in tqdm(ids[mod_name], desc="preload videos"):
                self.cached_data[id_] = self.load_from_file(id_)
                if debug_size < DBG_LOADER_LIMIT: # limit the dataloading for quick debugging
                    if i > debug_size:
                        break
                    i += 1

    def __getitem__(self, id_):
        if self.cached_data is None:
            return self.load_from_file(id_)
        else:
            return self.cached_data[id_]

    def load_from_file(self, id_):
        pck_file = os.path.join(self.pck_folder, 'id_' + str(id_) + '_feat.pickle')
        with open(pck_file, 'rb') as pickle_file:
             data = pck.load(pickle_file)
        return data

class LSMDCVideoPickleSaver:
    def __init__(
            self, dataset_path_dict: EasyDict, ids: List[str],
            preload: bool, data_split, debug_size: int=DBG_LOADER_LIMIT):
        self.h5_path = {}
        for i_mod in dataset_path_dict["video_feats"]:
            self.h5_path[i_mod] = dataset_path_dict["video_feats"][i_mod]
            
        self.cached_data = None
        self.features_source = "h5"
        self.data_keys = ids
        self.data_split = data_split
        self.save_folder =  dataset_path_dict["pickle_path"]
        # self.h5_path['object'] = "/mnt/efs/fs1/workspace/experiments/data/lsmdc16/debug/modality_experts"
        pck_obj_folder = os.path.join(self.save_folder, self.data_split)
        os.makedirs(pck_obj_folder, exist_ok=True)

        
        self.cached_data = {}
        h5_all = {}
        # buffer data to memory
        for i_mod in dataset_path_dict["video_feats"]:
            self.cached_data[i_mod] = {}
            h5_all[i_mod] = []
        for i_mod in dataset_path_dict["video_feats"]:
            h5_all[i_mod] = h5py.File(self.h5_path[i_mod], "r")
        i = 0

        for key in tqdm(self.data_keys[i_mod], desc="Preloading {} - modality: {} ==>".format(self.data_split, i_mod)):
            data_pck= {}
            for i_mod in dataset_path_dict["video_feats"]:
                print(i_mod, key)
                data = h5_all[i_mod][key]
                data_pck[i_mod] = general.make_shared_array(data)  

            gp_obj_file = os.path.join(pck_obj_folder, 'id_' + str(key) + '_feat.pickle')
            with open(gp_obj_file, 'wb') as f:    
                pck.dump(data_pck, f, pck.HIGHEST_PROTOCOL)
            if debug_size < DBG_LOADER_LIMIT: # limit the dataloading for quick debugging
                if i > debug_size:
                    break
                i += 1
                
    def __getitem__(self, id_):
        return self.cached_data[id_]               

class Youcook2VideoFeatureLoader:
    def __init__(
            self, dataset_path_dict: EasyDict, ids: List[str],
            preload: bool):
        self.h5_path = dataset_path_dict["video_feats"]
        self.cached_data = None
        if preload:
            self.cached_data = {}
            h5file = h5py.File(self.h5_path, "r")
            for id_ in tqdm(ids, desc="preload videos"):
                np_array = h5file[id_]
                shared_array = general.make_shared_array(np_array)
                self.cached_data[id_] = shared_array

    def __getitem__(self, id_):
        if self.cached_data is None:
            h5file = h5py.File(self.h5_path, "r")
            features = np.array(h5file[id_])
            h5file.close()
            return features
        else:
            return self.cached_data[id_]


class VideoDatasetFeatures(data.Dataset):
    def __init__(
            self, dataset_path_dict: EasyDict,
            split: str, max_frames: int, is_train: bool,
            preload_vid_feat: bool, preload_text_feat: bool,
            frames_noise: float, debug_size: int=DBG_LOADER_LIMIT, pickle_path=None):
        self.frames_noise = frames_noise
        self.split = split
        self.max_frames = max_frames
        self.is_train = is_train
        self.load_pickle = (pickle_path == "")
        self.debug_size = debug_size


        meta_file = dataset_path_dict["meta_data"]
        self.vids_dict = {}
        for i_meta, i_path in meta_file.items():
            json_file = i_path.open("rt", encoding="utf8")

            self.vids_dict[i_meta] = json.load(json_file,
                                object_pairs_hook=OrderedDict)
            json_file.close()
        
        self.ids = {}
        self.modalities = []
        # print(self.split)
        for i_mod, i_dict in self.vids_dict.items():
            self.modalities.append(i_mod)
            self.ids[i_mod] = [key for key, val in i_dict.items(
            ) if val["split"] == self.split]
            print("init modality {} of dataset {} split {} length {} ".format(i_mod, dataset_path_dict["dataset_name"], split, len(self.ids[i_mod])))


        if dataset_path_dict["dataset_name"] == "lsmdc16":
            self.preproc_par_fn = preprocess_bert_paragraph
            self.text_data = BertTextFeatureLoader(
                dataset_path_dict, self.ids, preload_text_feat, debug_size=self.debug_size)
            # self.vid_data = LSMDCVideoFeatureLoader(
            #     dataset_path_dict, self.ids, preload_vid_feat, self.split, debug_size=self.debug_size)
            if pickle_path == "":
                print("==> Start loading hdf5 files ... (Might be slower and needs more memory)")
                self.vid_data = LSMDCVideoFeatureLoader(
                dataset_path_dict, self.ids, preload_vid_feat, self.split, debug_size=self.debug_size)
            else:
                print("==> Start loading pickle files ...")
                self.vid_data = LSMDCVideoPickleLoader(
                    dataset_path_dict, self.ids, preload_vid_feat, self.split, debug_size=self.debug_size)
                
                
        elif dataset_path_dict["dataset_name"] == "youcook2":
            self.preproc_par_fn = preprocess_bert_paragraph
            self.text_data = BertTextFeatureLoader(
                dataset_path_dict, self.ids, preload_text_feat)
            self.vid_data = Youcook2VideoFeatureLoader(
                dataset_path_dict, self.ids, preload_vid_feat)
        else:
            raise NotImplementedError

    def get_frames_from_video(
            self, vid_id, indices=None, num_frames=None, modality_name: str = "action"):
        vid_dict = self.vids_dict[modality_name][vid_id]
        vid_len = vid_dict["num_frames"]
        if num_frames is not None:
            indices = general.compute_indices(
                vid_len, num_frames, self.is_train)

        if self.load_pickle:
            frames = self.vid_data[modality_name][vid_id][indices]
        else:
            frames = self.vid_data[vid_id][modality_name][indices]
        # 
        
        # print("frames: ", frames)
        return frames

    def get_frames_from_segment(
            self, vid_id, seg_num, num_frames, modality_name: str = "action"):
        vid_dict = self.vids_dict[modality_name][vid_id]
        seg = vid_dict["segments"][seg_num]
        start_frame = seg["start_frame"]
        seg_len = seg["num_frames"]
        indices = general.compute_indices(seg_len, num_frames, self.is_train)
        indices += start_frame
        frames = self.get_frames_from_video(vid_id, indices, modality_name=modality_name)
        return frames

    def __len__(self):
        if self.debug_size < DBG_LOADER_LIMIT:
            return self.debug_size
        else:
            return len(self.ids[self.modalities[0]])

    def __getitem__(self, index):
        
        get_data = {}
        for i_mod in self.modalities:
            get_data[i_mod]={}
            
            vid_id = self.ids[i_mod][index]
            vid_dict = self.vids_dict[i_mod][vid_id]
            clip_num = len(vid_dict["segments"])
            sent_num = len(vid_dict["segments"])

            # load video frames
            vid_frames_len = vid_dict["num_frames"]
            # print(i_mod, "vid_frames_len: ", vid_frames_len)
            if vid_frames_len > self.max_frames:
                vid_frames_len = self.max_frames
            vid_frames = torch.tensor(self.get_frames_from_video(
                vid_id, num_frames=vid_frames_len, modality_name=i_mod))
            vid_frames_len = int(vid_frames.shape[0])
            if self.frames_noise != 0:
                vid_frames_noise = general.truncated_normal_fill(
                    vid_frames.shape, std=self.frames_noise)
                vid_frames += vid_frames_noise

            # load segment frames
            clip_frames_list = []
            clip_frames_len_list = []
            for i, seg in enumerate(vid_dict["segments"]):
                c_num_frames = seg["num_frames"]
                # print("num_frames: ", c_num_frames)
                if c_num_frames > self.max_frames:
                    c_num_frames = self.max_frames
                c_frames = self.get_frames_from_segment(
                    vid_id, i, num_frames=c_num_frames, modality_name=i_mod)
                c_frames = torch.tensor(c_frames)
                if self.frames_noise != 0:
                    clip_frames_noise = general.truncated_normal_fill(
                        c_frames.shape, std=self.frames_noise)
                    c_frames += clip_frames_noise
                clip_frames_list.append(c_frames)
                clip_frames_len_list.append(c_frames.shape[0])
                # print(clip_frames_len_list)
                # print("-0-"*20)

            # load text
            seg_narrations = []
            for seg in vid_dict["segments"]:
                seg_narr = seg["narration"]
                if seg_narr is None:
                    seg_narr = "undefined"
                    print("WARNING: Undefined text tokens "
                        "(no narration data, is this a test set?)")
                seg_narrations.append(seg_narr)
            list_of_list_of_words = self.preproc_par_fn(seg_narrations)

            # load precomputed text features
            par_cap_vectors, sent_cap_len_list = self.text_data[vid_id]
            par_cap_len = int(par_cap_vectors.shape[0])
            par_cap_vectors = torch.tensor(par_cap_vectors).float()

            # split paragraph features into sentences
            sent_cap_vectors_list = []
            pointer = 0
            for i, sent_cap_len in enumerate(sent_cap_len_list):
                sent_cap_vectors = par_cap_vectors[
                                pointer:pointer + sent_cap_len, :]
                sent_cap_vectors_list.append(sent_cap_vectors)
                pointer += sent_cap_len

            # print(vid_id, "====>")
            get_data[i_mod]={
                "vid_id": vid_id,
                "data_words": list_of_list_of_words,
                "vid_frames": vid_frames,
                "vid_frames_len": vid_frames_len,
                "par_cap_vectors": par_cap_vectors,
                "par_cap_len": par_cap_len,
                "clip_num": clip_num,
                "sent_num": sent_num,
                "clip_frames_list": clip_frames_list,
                "clip_frames_len_list": clip_frames_len_list,
                "sent_cap_len_list": sent_cap_len_list,
                "sent_cap_vectors_list": sent_cap_vectors_list
            }

        return get_data


    def collate_fn(self, data_batch):
        def get_data(i_mod, key):
            return [d[i_mod][key] for d in data_batch]

        batch_size = len(data_batch)
        batch_data = {}
        for i_mod in self.modalities:
        # collate video frames
            batch_data[i_mod] = {}
            list_vid_frames = get_data(i_mod, "vid_frames")
            list_vid_frames_len = get_data(i_mod, "vid_frames_len")
            vid_feature_dim = list_vid_frames[0].shape[-1]
            vid_frames_len = torch.tensor(list_vid_frames_len).long()
            vid_frames_max_seq_len = int(vid_frames_len.max().numpy())
            vid_frames = torch.zeros(
                batch_size, vid_frames_max_seq_len, vid_feature_dim).float()
            vid_frames_mask = torch.zeros(batch_size, vid_frames_max_seq_len)
            for batch, (seq_len, item) in enumerate(zip(
                    list_vid_frames_len, list_vid_frames)):
                vid_frames[batch, :seq_len] = item
                vid_frames_mask[batch, :seq_len] = 1

            # collate paragraph features
            list_par_cap_len = get_data(i_mod, "par_cap_len")
            list_par_cap_vectors = get_data(i_mod, "par_cap_vectors")
            par_feature_dim = list_par_cap_vectors[0].shape[-1]
            par_cap_len = torch.tensor(list_par_cap_len).long()
            par_cap_max_len = int(par_cap_len.max().numpy())
            par_cap_vectors = torch.zeros(
                batch_size, par_cap_max_len, par_feature_dim).float()
            par_cap_mask = torch.zeros(batch_size, par_cap_max_len)
            for batch, (seq_len, item) in enumerate(
                    zip(list_par_cap_len, list_par_cap_vectors)):
                par_cap_vectors[batch, :seq_len, :] = item
                par_cap_mask[batch, :seq_len] = 1

            # collate clip frames
            list_clip_num = get_data(i_mod, "clip_num")
            clip_num = torch.tensor(list_clip_num).long()
            total_clip_num = int(np.sum(list_clip_num))
            list_clip_frames_len_list = get_data(i_mod, "clip_frames_len_list")
            clip_frames_max_len = int(np.max(
                [np.max(len_single) for len_single in list_clip_frames_len_list]))
            clip_frames = torch.zeros((
                total_clip_num, clip_frames_max_len, vid_feature_dim)).float()
            clip_frames_mask = torch.zeros(
                (total_clip_num, clip_frames_max_len))
            list_clip_frames_list = get_data(i_mod, "clip_frames_list")
            clip_frames_len = []
            c_num = 0
            for batch, clip_frames_list in enumerate(list_clip_frames_list):
                for i, clip_frames_item in enumerate(clip_frames_list):
                    clip_frames_len_item = int(clip_frames_item.shape[0])
                    clip_frames[c_num, :clip_frames_len_item, :] =\
                        clip_frames_item
                    clip_frames_mask[c_num, :clip_frames_len_item] = 1
                    clip_frames_len.append(clip_frames_len_item)
                    c_num += 1
            clip_frames_len = torch.tensor(clip_frames_len).long()

            # collate sentence features
            list_sent_num = get_data(i_mod, "sent_num")
            sent_num = torch.tensor(list_sent_num).long()
            total_sent_num = int(np.sum(list_sent_num))
            list_sent_cap_len_list = get_data(i_mod, "sent_cap_len_list")
            sent_cap_max_len = int(np.max(
                [np.max(len_single) for len_single in list_sent_cap_len_list]))
            sent_cap_len = []
            sent_cap_mask = torch.zeros(
                (total_sent_num, sent_cap_max_len)).long()
            cap_feature_dim = list_par_cap_vectors[0].shape[-1]
            sent_cap_vectors = torch.zeros(
                (total_sent_num, sent_cap_max_len, cap_feature_dim))
            c_num = 0
            for batch, sent_cap_len_list in enumerate(
                    list_sent_cap_len_list):
                pointer = 0
                for sent_cap_len_item in sent_cap_len_list:
                    sent_cap_vectors[c_num, :sent_cap_len_item] =\
                        par_cap_vectors[
                        batch, pointer:pointer + sent_cap_len_item]
                    sent_cap_mask[c_num, :sent_cap_len_item] = 1
                    sent_cap_len.append(sent_cap_len_item)
                    c_num += 1
                    pointer += sent_cap_len_item
            sent_cap_len = torch.tensor(sent_cap_len).long()

            batch_data[i_mod] = {
                "vid_frames": vid_frames,
                "vid_frames_mask": vid_frames_mask,
                "vid_frames_len": vid_frames_len,
                "par_cap_vectors": par_cap_vectors,
                "par_cap_mask": par_cap_mask,
                "par_cap_len": par_cap_len,
                "clip_num": clip_num,
                "clip_frames": clip_frames,
                "clip_frames_len": clip_frames_len,
                "clip_frames_mask": clip_frames_mask,
                "sent_num": sent_num,
                "sent_cap_vectors": sent_cap_vectors,
                "sent_cap_mask": sent_cap_mask,
                "sent_cap_len": sent_cap_len,
                "vid_id": get_data(i_mod, "vid_id"),
                "data_words": get_data(i_mod, "data_words")
            }
        return batch_data


def create_datasets(
        dataset_path_dict: EasyDict, cfg: EasyDict, preload_vid_feat: bool,
        preload_text_feat: bool, eval=False, test=False,
         debug_train_size: int=DBG_LOADER_LIMIT, debug_val_size: int=DBG_LOADER_LIMIT, 
         debug_test_size: int=DBG_LOADER_LIMIT, pickle_path=None):
    
    if eval:
        val_set = VideoDatasetFeatures(dataset_path_dict,
        cfg.dataset.val_split, cfg.dataset.max_frames, False, preload_vid_feat,
        preload_text_feat, 0, pickle_path=pickle_path)
        return val_set

    if test:
        test_set = VideoDatasetFeatures(dataset_path_dict,cfg.dataset.test_split, cfg.dataset.max_frames,
         False, preload_vid_feat,preload_text_feat, 0, debug_size=debug_test_size, pickle_path=pickle_path)
        return test_set

    # print("Train loader", "00"*20)
    train_set = VideoDatasetFeatures(dataset_path_dict,
        cfg.dataset.train_split, cfg.dataset.max_frames, True,
        preload_vid_feat, preload_text_feat, cfg.dataset.frames_noise, debug_size=debug_train_size, pickle_path=pickle_path)
    # print("Val loader", "00"*20)
    val_set = VideoDatasetFeatures(dataset_path_dict,
        cfg.dataset.val_split, cfg.dataset.max_frames, False, preload_vid_feat,
        preload_text_feat, 0, debug_size=debug_val_size, pickle_path=pickle_path)
    return train_set, val_set


def create_loaders(
        train_set: VideoDatasetFeatures, val_set: VideoDatasetFeatures, test_set: VideoDatasetFeatures,
        batch_size: int, num_workers: int, eval=False):
    if eval:
        if val_set is not None:
            val_loader = data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=val_set.collate_fn,
            pin_memory=True)
            return val_loader
        if test_set is not None:
            test_loader = data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=test_set.collate_fn,
            pin_memory=True)
            return test_loader
    # train_loader = data.DataLoader(
    #     train_set, batch_size=batch_size, shuffle=True,
    #     num_workers=num_workers, collate_fn=train_set.collate_fn,
    #     pin_memory=True)
    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=train_set.collate_fn,
        pin_memory=True)
    val_loader = data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
         collate_fn=val_set.collate_fn,
         pin_memory=True)
    if test_set is not None:
        test_loader = data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            collate_fn=val_set.collate_fn,
             pin_memory=True)    
    else:
        test_loader = None
    return train_loader, val_loader, test_loader
