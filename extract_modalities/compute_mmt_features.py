import argparse
from collections import OrderedDict
from copy import deepcopy

import h5py
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn

from thirdparty.s3dg import S3D
from utils_collection import write_json, breakpoint
from utils_collection.lsmdc_annot import build_vocab, build_label
from extract_modalities.mmt_modalities import get_sample_data
import pdb
import json


@torch.no_grad()
def main():
    """main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--num_cuda", type=int, default=1)
    parser.add_argument("--modality_name", nargs="+", default=["mmt"])
    parser.add_argument("--data_save",
                        type=str,
                        default="data",
                        help="save path for features and meta data")
    parser.add_argument("--data_video",
                        type=str,
                        default="data",
                        help="path to dataset video")

    parser.add_argument("--group_k", type=int, default=5, help="Number of segments per video")
    parser.add_argument("--output_file", type=str, default="lsmdc_v1.pth")
    args = parser.parse_args()

    params = {}
    params['word_count_threshold'] = 5
    params['input_path'] = 'data/lsmdc/annot/'
    params['group_by'] = args.group_k
    params['max_length'] = 30
    params['annot'] = ['LSMDC16_annos_training.csv']#, 'LSMDC16_annos_test.csv']
    params['splits'] = ['train'] #, 'test']
    # params['annot'] = ['LSMDC16_challenge_1000_publictect.csv']#, 'LSMDC16_annos_test.csv']
    # params['splits'] = ['test1k'] #, 'test']

    videos, groups, movie_ids, vocab = build_vocab(params)
    itow = {i + 2: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 2 for i, w in enumerate(vocab)}  # inverse table
    wtoi['<eos>'] = 0
    itow[0] = '<eos>'
    wtoi['<sos>'] = 1
    itow[1] = '<sos>'

    labels, lld = build_label(params, videos, wtoi)


    #==================================================================================
    # # load model
    # net = S3D(token_to_word_path="data/howto100m_mil/s3d_dict.npy",
    #           num_classes=512)
    # net.load_state_dict(torch.load('data/howto100m_mil/s3d_howto100m.pth'))
    # net = net.eval()
    # if args.cuda:
    #     net = net.cuda()


    print(f"{len(groups)} vids {len(videos)} segs")

    # make frame features

        # create directories to store features
    data_dir = os.path.join(args.data_save, "group{}".format(args.group_k),
                            "video_features")
    language_dir = os.path.join(args.data_save, "group{}".format(args.group_k),
                                "language_features")
    meta_dir = os.path.join(args.data_save, "meta")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(language_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    modality_name = '_'.join(args.modality_name).replace(',', '')
    # print(modality_name)
    vid_h5_file = Path(os.path.join(data_dir, modality_name) + '.h5')
    print(vid_h5_file)
    vid_h5 = h5py.File(vid_h5_file, "a")
    meta_data = OrderedDict()
    pbar = tqdm(total=len(groups))


    rnd_idx = np.random.randint(20000, size=(1, 100))
    groups_rnd = [groups[i] for i in rnd_idx[0]]

    i = 0
    for vid_gp in groups_rnd:
        movie_name = vid_gp["movie"]
        clip_ids = vid_gp["videos"]
        vid_gp_id = vid_gp['id']

        results_collector = []
        start_frame = 0
        stop_frame = 0
        clip_counter = 0
        meta_video = OrderedDict()
        video_segments = []
        
        if str(vid_gp_id) not in vid_h5:

            for clip_id in clip_ids:
                clip_name = videos[clip_id]['clip']
                clip_caption = videos[clip_id]['narration']
                split = vid_gp["split"]

                data_dir = "/mnt/efs/fs1/workspace/experiments/data_mmt_eccv20/mmt/data/LSMDC/vid_feat_files/mult_h5/"
                expert_names = ["scene", "rgb", "ocr","s3d","vggish", "audio", "flow"]
                #print(clip_name, movie_name)
                raw_captions, raw_captions_t, features, features_t, features_avgpool, features_maxpool = get_sample_data(data_dir, clip_name, expert_names)
                #print(features)
                if features["scene"].shape[1] > 2:
                    scene_feat1 = features["scene"][0,0:1024].reshape(1,1024)
                    scene_feat2 = features["scene"][0,1024:2048].reshape(1,1024)
                else:
                    print("missing scene", clip_name)
                    scene_feat1 = np.zeros((1,1024))
                    scene_feat2 = np.zeros((1,1024))
                
                if features["rgb"].shape[1] > 2:
                    rgb_feat1 = features["rgb"][0,0:1024].reshape(1,1024)
                    rgb_feat2 = features["rgb"][0,1024:2048].reshape(1,1024)
                else:
                    print("missing RGB", clip_name)
                    rgb_feat1 = np.zeros((1,1024))
                    rgb_feat2 = np.zeros((1,1024))
                
                if features["s3d"].shape[1] > 2:
                    s3d_feat1 = features["s3d"][0,:].reshape(1,1024)
                    s3d_feat2 = features["s3d"][1,:].reshape(1,1024)
                else:
                    print("missing S3D", clip_name)
                    s3d_feat1 = np.zeros((1,1024))
                    s3d_feat2 = np.zeros((1,1024))
                
                if features["flow"].shape[1] > 2:
                    flow_feat = features["flow"][0,:].reshape(1,1024)
                else:
                    print("missing Flow", clip_name)
                    flow_feat = np.zeros((1,1024))

                if features["ocr"].shape[1] > 300:
                    ocr_feat = features["ocr"].reshape(1,-1)
                else:
                    print("missing OCR", clip_name)
                    ocr_feat = np.zeros((1,600))
                    
                if features["vggish"].shape[1] > 2:
                    vggish_feat = features["vggish"].reshape(1,-1)
                else:
                    print("missing vggish",clip_name)
                    vggish_feat = np.zeros((1,384))
                
                if features["audio"].shape[1] > 2:
                    audio_feat = features["audio"].reshape(1,-1)
                else:
                    print("missing audio", clip_name)
                    audio_feat = np.zeros((1,40))
                # print(features["vggish"].shape, vggish_feat.shape, "===>")

                
                # feat_all = torch.cat(features_all, dim=0)
                ocr_vggish_feat = np.zeros((1,1024))
                ocr_vggish_feat[0, 0:600] = ocr_feat[0, 0:600]
                ocr_vggish_feat[0, 600:984] = vggish_feat[0, 0:384]
                ocr_vggish_feat[0, 984:1024] = audio_feat[0,0:40]
                expert_feats = [scene_feat1, scene_feat2, rgb_feat1, rgb_feat2, s3d_feat1, s3d_feat2, ocr_vggish_feat, flow_feat]
                results_collector.append(expert_feats)
                # pdb.set_trace()
                video_path = os.path.join(args.data_video, movie_name,
                                          clip_name) + '.mp4'


                # Segment level information
                #tmp_feat = torch.cat(results_collector, dim=0)
                start_frame = stop_frame 
                stop_frame = len(results_collector)*8
                num_features = stop_frame - start_frame 
                # print(start_frame, stop_frame, num_features, len(results_collector))
                segment_info = {
                    "narration": clip_caption,
                    "start_frame": start_frame,
                    "stop_frame": stop_frame,
                    "segment_name": clip_name,
                    "num_frames": num_features
                }

                video_segments.append(segment_info)
                clip_counter += 1
            results_feat = np.concatenate(results_collector, axis=0).reshape(-1, 1024)
            # pdb.set_trace()
            len_results = results_feat.shape[0]
            # Video level information
            meta_video = {
                "num_frames": len_results,
                "data_id": str(vid_gp_id),
                "split": split,
                "segments": video_segments
            }

            # write to h5
            vid_h5[str(vid_gp_id)] = results_feat
            feat_len = len_results
            # del results
            # del frames_collector
            del results_collector
            del results_feat
            # del video_decord
        else:
            feat_len = int(vid_h5[str(vid_gp_id)].shape[0])

        # write new meta
        meta_data[str(vid_gp_id)] = meta_video

        pbar.update()
    vid_h5.close()
    pbar.close()

    # write new meta

    meta_data_path = Path(
        os.path.join(
            args.data_save, "meta",
            "meta_group{}_{}.json".format(args.group_k, modality_name)))

    with open(meta_data_path, "wt") as fh:
        write_json(meta_data, fh)


    print("=====Done!=====")
if __name__ == '__main__':
    main()
