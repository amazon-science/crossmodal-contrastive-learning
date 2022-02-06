"""
Code to load expert features provided by:
MMT: Multi-modal Transformer for Video Retrieval
https://github.com/gabeur/mmt.git

"""
import os
import h5py

import ipdb
import numpy as np

# Dict used to calculate at which moment of the video was each feature extracted
expert_timings = {
    'flow': {
        'feat_width': None,
    },
    'audio': {
        'feat_width': None,
    },
    'rgb': {
        'feat_width': 0.2,
    },
    'face': {
        'feat_width': None,
    },
    'scene': {
        'feat_width': 1.0,
    },
    'speech': {
        'feat_width': None,
    },
    'ocr': {
        'feat_width': None,
    },
    's3d': {
        'feat_width': 1.0,
    },
    'vggish': {
        'feat_width': 1.0,
    },
    'audio_c': {
        'feat_width': None,
    },
    'face_c': {
        'feat_width': None,
    },
    'ocr_c': {
        'feat_width': None,
    },
    'speech_c': {
        'feat_width': None,
    },
}

def get_sample_data(data_dir, vid_name, expert_names):

    output_basename = f"{vid_name[0]}/{vid_name[1]}/{vid_name[2]}/{vid_name}.h5"
    dataset_file_path = os.path.join(data_dir, output_basename)

    dataset_file = h5py.File(dataset_file_path, "r")
    with h5py.File(dataset_file_path, "r") as dataset_file:
        video_data = dataset_file
        keys_list = list(video_data.keys())
        nb_captions = len(
            [k for k in keys_list if k.startswith("raw_captions.")])
        # if nb_captions == 0:
        #    logger.warning("No caption for %s", dataset_file_path)
        # assert nb_captions > 0
        raw_captions = []
        raw_captions_t = []
        for i in range(nb_captions):
            raw_caption = video_data[f"raw_captions.{i}"].value
            raw_captions.append(raw_caption)
            if f"raw_captions_t.{i}" in video_data.keys():
                raw_caption_t = video_data[f"raw_captions_t.{i}"].value
                if raw_caption_t.shape[0] != len(raw_caption):
                   raw_caption_t = raw_caption_t[:len(raw_caption)]
                raw_captions_t.append(raw_caption_t)
            else:
                nb_words = len(raw_caption)
                raw_caption_t = np.zeros((nb_words, 2))
                raw_captions_t.append(raw_caption_t)

        features = {}
        features_t = {}
        features_avgpool = {}
        features_maxpool = {}
        for expert in expert_names:
            # print(video_data.keys(), "------>")
            features[expert] = np.zeros((1,2))
            if f"features.{expert}" in video_data.keys():
                # print("<====computer for {} expert ==== >".format(expert))
                x = video_data[f"features.{expert}"].value

                if len(x) > 0 and not np.isnan(x[0][0]):
                   features[expert] = video_data[f"features.{expert}"].value
                   nb_feats = features[expert].shape[0]
                else:
                   features[expert] = np.zeros((1,2))
                   nb_feats = 1

                if f"features_t.{expert}" in video_data.keys():
                    x = video_data[f"features_t.{expert}"].value
                    # if not np.isnan(x[0][0]):
                    if expert in ["s3d", "vggish"]:
                       features_t[expert] = video_data[
                        f"features_t.{expert}"].value
                    if features_t[expert].shape[0] != features[expert].shape[0]:
                        # logger.warning(
                        #     "Incorrect number of features_t values "
                        #     "for %s", dataset_file_path)
                        features_t[expert] = features_t[expert][:features[expert].
                                                                shape[0]]
                    else:
                        # nb_feats = features[expert].shape[0]
                        expert_timing = expert_timings[expert]
                        features_t[expert] = get_feature_timings(
                            nb_feats, **expert_timing)
                else:
                    # nb_feats = features[expert].shape[0]
                    #print(features[expert].shape)
                    expert_timing = expert_timings[expert]
                    features_t[expert] = get_feature_timings(
                        nb_feats, **expert_timing)
                features_t[expert] = np.average(features_t[expert], axis=1)
            features_avgpool[expert] = None
            features_maxpool[expert] = None
    return (raw_captions, raw_captions_t, features, features_t,
            features_avgpool, features_maxpool)


def get_feature_timings(nb_feats, feat_width, stride=None, group=None):
    # Return an array containing the start time of each feature in the first
    # line and the end time of each feature in the second line.
    if feat_width is None:
      timings = np.empty((nb_feats, 2))
      timings[:] = -1
      return timings
    if group is not None:
      assert nb_feats % group == 0
      nb_feats_top = nb_feats // group
      top_timings = get_feature_timings(nb_feats_top,
                                             feat_width,
                                             stride,
                                             group=None)
      bot_timings = np.repeat(top_timings, group, axis=-1)
      return bot_timings
    if stride is None:
      stride = feat_width
    starts = np.linspace(0, (nb_feats - 1) * stride, num=nb_feats)
    ends = np.linspace(feat_width, (nb_feats - 1) * stride + feat_width,
                       num=nb_feats)
    res = np.stack((starts, ends), axis=-1)
    return res