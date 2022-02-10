import argparse
from collections import OrderedDict
from copy import deepcopy
import decord
from decord import cpu
import h5py
import numpy as np
import os
from pathlib import Path
import pickle as pk
from tqdm import tqdm
import librosa

import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

from thirdparty.s3dg import S3D
from utils_collection import write_json, breakpoint
from utils_collection.lsmdc_annot import build_vocab, build_label
import pdb
import json

import thirdparty.action_ircsn_res152.video_transforms as video_transforms
import thirdparty.action_ircsn_res152.volume_transforms as volume_transforms
from thirdparty.action_ircsn_res152.ircsnv2 import ircsn_v2_resnet152_f32s2_kinetics400
from thirdparty.appearance_resnest269.resnest import resnest101, resnest200, resnest269
from thirdparty.scene_densenet161.densenet import densenet161
from thirdparty.audio_resnet50.resnet import resnet50
from thirdparty.object_faster_rcnn_resnet50_fpn.faster_rcnn import fasterrcnn_resnet50_fpn
from thirdparty.face_mtcnn_facenet.mtcnn import MTCNN
from thirdparty.face_mtcnn_facenet.inception_resnet_v1 import InceptionResnetV1
from thirdparty.flow_pwcnet.pwcnet import Network as pwcnet


def prepare_modality_networks(modality_list):
    modality_models = {}
    ext_feature = True

    # Action
    if 'action' in modality_list:
        model_action = ircsn_v2_resnet152_f32s2_kinetics400(
            extract_feat=ext_feature)
        model_action.eval()
        print("===> Action model loaded")
        modality_models['action'] = model_action.cuda()

    if 'flow' in modality_list:
        model_flow = pwcnet().cuda().eval()
        model_pwc = torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + 'default' + '.pytorch', file_name='pwc-' + 'default', model_dir='data/zhu_modalities/')
        model_flow.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in model_pwc.items() })
        print("===> Flow model loaded")
        modality_models['flow'] = model_flow

    # Appearance
    if 'appearance' in modality_list:
        model_appearance = resnest269(extract_feat=ext_feature)
        model_appearance.eval()
        print("===> Appearance model loaded")
        modality_models['appearance'] = model_appearance.cuda()

    # Scene
    if 'scene' in modality_list:
        classes = list()
        model_file = 'data/zhu_modalities/densenet161_places365.pth.tar'

        model_scene = densenet161(num_classes=365, extract_feat=ext_feature)
        checkpoint_scene = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        state_dict = {
            str.replace(k, 'module.', ''): v
            for k, v in checkpoint_scene['state_dict'].items()
        }
        state_dict = {
            str.replace(k, 'norm.', 'norm'): v
            for k, v in state_dict.items()
        }
        state_dict = {
            str.replace(k, 'conv.', 'conv'): v
            for k, v in state_dict.items()
        }
        state_dict = {
            str.replace(k, 'normweight', 'norm.weight'): v
            for k, v in state_dict.items()
        }
        state_dict = {
            str.replace(k, 'normrunning', 'norm.running'): v
            for k, v in state_dict.items()
        }
        state_dict = {
            str.replace(k, 'normbias', 'norm.bias'): v
            for k, v in state_dict.items()
        }
        state_dict = {
            str.replace(k, 'convweight', 'conv.weight'): v
            for k, v in state_dict.items()
        }
        model_scene.load_state_dict(state_dict)
        model_scene.eval()
        print("===> Scene model loaded")
        modality_models['scene'] = model_scene.cuda()

    # Audio
    if 'audio' in modality_list:
        # get audio model
        model_audio = resnet50()
        msg = model_audio.load_state_dict(
            torch.load('data/zhu_modalities/audionet_torchvision.pth'),
            strict=False)
        assert set(msg.missing_keys) == {'fc.weight', 'fc.bias'}
        model_audio.eval()
        modality_models['audio'] = model_audio.cuda()
        print("===> Audio finetuned model loaded")

    # HowTo100M
    if 'howto100m' in modality_list:
        model_s3d = S3D(token_to_word_path="data/howto100m_mil/s3d_dict.npy",
                        num_classes=512)
        model_s3d.load_state_dict(
            torch.load('data/howto100m_mil/s3d_howto100m.pth'))
        model_s3d = model_s3d.eval().cuda()
        print('===> HowTo100m base model loaded')
        modality_models['howto100m'] = model_s3d.cuda()

    # HowTo100M - finetune LSMDC
    if 'howto100m_finetune' in modality_list:
        model_s3d_finetune = S3D(
            token_to_word_path="data/howto100m_mil/s3d_dict.npy",
            num_classes=512)
        # model_s3d_finetune = torch.nn.DataParallel(model_s3d_finetune)
        checkpoint = torch.load(
            'data/howto100m_mil/s3d_howto100m_finetune_v2.pth.tar'
        )["state_dict"]
        checkpoint_module = {}
        for k, v in checkpoint.items():
            if "_module." not in k:
                checkpoint_module[k.replace('module.', '')] = v
            if "_module." in k:
                checkpoint_module[k.replace('module.text', 'text')] = v

        model_s3d_finetune.load_state_dict(checkpoint_module)
        model_s3d_finetune = model_s3d_finetune.eval()
        model_s3d_finetune = model_s3d_finetune.cuda()
        print("===> HowTo100M finetuned model loaded")
        modality_models['howto100m_finetune'] = model_s3d_finetune

    # Object
    if 'object' in modality_list:
        # fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
        model_object = fasterrcnn_resnet50_fpn(pretrained=False)
        model_object.load_state_dict(
            torch.load('data/zhu_modalities/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'),
            strict=False)
        model_object.eval()
        print("===> Object model loaded")
        modality_models['object'] = model_object.cuda()


    # Face
    if 'face' in modality_list:
            # Get MTCNN model for face detection
        model_face = MTCNN()
        model_face.eval()
        print("===> Face model loaded")
        modality_models['face'] = model_face.cuda()

    return modality_models


def clip_sliding(frames, model):
    clip_features = []

    def feed_batch(input_list):
        batch_input = torch.stack(input_list, dim=0)
        batch_input = batch_input.float() / 255
        #print("batch input shape: ", batch_input.shape)
        # if args.cuda:
        batch_input = batch_input.cuda()
        # if args.cuda and args.num_cuda > 1:
        #     result = nn.parallel.data_parallel(
        #         model, batch_input,
        #         range(args.num_cuda))["video_embedding"]
        # else:
        result = model(batch_input, '', mode='video')  #["video_embedding"]
        #print("result shape: ", result.shape)
        return result

    # given some number of frames, get 16 frames with stride 4
    batch_size = 8
    sliding_window_size = 16
    stride = 2
    frames_collector = []
    num_frames = len(frames)

    for pointer in range(0, num_frames, stride):
        frames_single = frames[:, pointer:pointer + sliding_window_size, :, :]

        if frames_single.shape[1] == sliding_window_size:
            # last frames are less than sliding_window_size. drop them?
            frames_collector.append(frames_single)
        if len(frames_collector) == batch_size:
            clip_features.append(feed_batch(frames_collector))
            frames_collector = []

    if len(frames_collector) > 0:
        clip_features.append(feed_batch(frames_collector))

    return clip_features


@torch.no_grad()
def main():
    """main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--num_cuda", type=int, default=1)
    parser.add_argument("--modality_list", nargs="+", default=["howto100m"])
    parser.add_argument("--data_save",
                        type=str,
                        default="data",
                        help="save path for features and meta data")
    parser.add_argument("--data_video",
                        type=str,
                        default="data",
                        help="path to dataset video")

    parser.add_argument("--group_k",
                        type=int,
                        default=5,
                        help="Number of segments per video")
    parser.add_argument("--output_file", type=str, default="lsmdc_v1.pth")
    args = parser.parse_args()

    params = {}
    params['word_count_threshold'] = 5
    params['input_path'] = 'data/lsmdc/annot_new/'
    params['group_by'] = args.group_k
    params['max_length'] = 50
    params['annot'] = ['LSMDC16_challenge_1000_publictect.csv'] #['LSMDC16_annos_training.csv', 'LSMDC16_annos_val.csv'] #['LSMDC16_challenge_1000_publictect.csv'] #
    params['splits'] = ['test1k']  #['train', 'val'] #['test1k'] #

    videos, groups, movie_ids, vocab = build_vocab(params)
    itow = {i + 2: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 2 for i, w in enumerate(vocab)}  # inverse table
    wtoi['<eos>'] = 0
    itow[0] = '<eos>'
    wtoi['<sos>'] = 1
    itow[1] = '<sos>'

    labels, lld = build_label(params, videos, wtoi)

    short_side_size = 256
    crop_size = 224
    transform_fn = video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(crop_size, crop_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    ])

    avg_pool2d = nn.AvgPool2d((7, 7))

    #==================================================================================
    # load model

    expert_models = prepare_modality_networks(args.modality_list)

    print(f"{len(groups)} vids {len(videos)} segs")

    # create directories to store features
    data_dir = os.path.join(args.data_save, "group{}".format(args.group_k))
    language_dir = os.path.join(args.data_save, "group{}".format(args.group_k))
    meta_dir = os.path.join(args.data_save, "meta")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(language_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    modality_name = '_'.join(args.modality_list).replace(',', '')
    # print(modality_name)
    vid_h5_file = Path(os.path.join(data_dir, modality_name) + '.h5')
    print(vid_h5_file)
    vid_h5 = h5py.File(vid_h5_file, "a")
    meta_data = OrderedDict()
    pbar = tqdm(total=len(groups))

    i = 0
    count_no_face = 0
    for vid_gp in groups:

        movie_name = vid_gp["movie"]
        clip_ids = vid_gp["videos"]
        vid_gp_id = vid_gp['id']

        results_collector = []
        start_frame = 0
        stop_frame = 0
        clip_counter = 0
        meta_video = OrderedDict()
        video_segments = []
        group_objects = []
        group_faces = []
        group_object_class = []

        if str(vid_gp_id) not in vid_h5:

            for clip_id in clip_ids:
                clip_name = videos[clip_id]['clip']
                clip_caption = videos[clip_id]['narration']
                split = vid_gp["split"]

                video_path = os.path.join(args.data_video, movie_name,
                                          clip_name) + '.mp4'
                
                # print(video_path)
                # video_path = 'thirdparty/action_ircsn_res152/4eWzsx1vAi8.mp4'
                # video_path = 'thirdparty/action_ircsn_res152/0001_American_Beauty_1.mp4'
                # decord.bridge.set_bridge('torch')
                if not os.path.isfile(video_path):
                    print(video_path)
                    continue

                if 'howto100m' in args.modality_list or 'howto100m_finetune' in args.modality_list:
                    decord.bridge.set_bridge('torch')
                    video_decord = decord.VideoReader(video_path,
                                                      ctx=cpu(0),
                                                      width=224,
                                                      height=224,
                                                      num_threads=10)

                    #STATS: mean len of clips= ~100 frames
                    frame_indices = range(0, len(video_decord), 2)

                    frames = video_decord.get_batch(frame_indices)
                    num_frames = len(frames)
                    frames = frames.permute(3, 0, 1, 2)

                else:
                    video_decord = decord.VideoReader(video_path,
                                                      num_threads=10)

                    #pdb.set_trace()
                    #STATS: mean len of clips= ~100 frames
                    frame_indices = [len(video_decord) // 2, (len(video_decord) // 2) + 1]
                    # print("===>", len(video_decord))
                    frames = video_decord.get_batch(frame_indices).asnumpy()
                    num_frames = len(frames)

                # Action
                # clip shape : [3, 32, 224, 224]
                if 'action' in args.modality_list:
                    action_features = []
                    frame_indices = range(0,16)
                    frames = video_decord.get_batch(frame_indices).asnumpy()
                    # print("===>", frames.shape)
                    clip_input_action = transform_fn(frames)
                    # print(clip_input.shape)

                    batch_size = 16
                    sliding_window_size = 16
                    stride = 2
                    frames_collector = []
                    # print("num frames: ", num_frames)
                    # print(pointer, sliding_window_size, clip_input.shape)
                    frames_single = torch.unsqueeze(clip_input_action, dim=0) #[:, pointer:pointer +
                                                #   sliding_window_size, :, :]


                    # batch_input = torch.stack(frames_single, dim=0)
                    # print(frames_single.shape)
                    feat = expert_models['action'](frames_single.cuda())
                    # print('The input video clip is classified to be class %d' % (np.argmax(feat[0,:])))

                    # print(feat.shape)
                    action_features.append(feat)

                    # print("len clip: ", len(clip_features))

                    # print('The input video clip is classified to be class %d' % (np.argmax(feat)))

                ## Appearance - Resnet
                if 'appearance' in args.modality_list:
                    appearance_features = []
                    num_samples = 1
                    clip_input = transform_fn(frames)
                    tick = num_frames / float(num_samples)
                    offsets = np.array([
                        int(tick / 2.0 + tick * x) for x in range(num_samples)
                    ])

                    for i_frm in offsets:
                        # print(clip_input.shape)
                        input_batch = torch.unsqueeze(
                            clip_input[:, i_frm, :, :], dim=0
                        )  # create a mini-batch as expected by the model
                        # print(input_batch.shape)
                        feat = expert_models['appearance'](input_batch.cuda())
                        # feat dim = 1 * 256
                        appearance_features.append(feat)
                    # print(feat.shape)
                    # print('The input image is classified to be class %d' % (np.argmax(feat)))


                # Scene
                if 'scene' in args.modality_list:
                    scene_features = []
                    num_samples = 1
                    clip_input = transform_fn(frames)
                    tick = num_frames / float(num_samples)
                    offsets = np.array([
                        int(tick / 2.0 + tick * x) for x in range(num_samples)
                    ])
                    for i_frm in offsets:
                        input_batch = torch.unsqueeze(
                            clip_input[:, i_frm, :, :], dim=0
                        )  # create a mini-batch as expected by the model
                        feat = expert_models['scene'].forward(
                            input_batch.cuda())
                        #print(feat.shape)
                        scene_features.append(feat)

                ## Appearance 
                if 'object' in args.modality_list:
                    COCO_INSTANCE_CATEGORY_NAMES = [
                        '__background__', 'person', 'bicycle', 'car',
                        'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'N/A',
                        'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                        'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
                        'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite',
                        'baseball bat', 'baseball glove', 'skateboard',
                        'surfboard', 'tennis racket', 'bottle', 'N/A',
                        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                        'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
                        'remote', 'keyboard', 'cell phone', 'microwave',
                        'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
                        'book', 'clock', 'vase', 'scissors', 'teddy bear',
                        'hair drier', 'toothbrush'
                    ]

                    object_features = []
                    clip_objects = []
                    clip_object_class = []
                    # clip_object_featues = []
                    num_samples = 1
                    short_side_size = 256
                    crop_size = 224
                    transform_fn_obj = video_transforms.Compose([
                        video_transforms.Resize(short_side_size, interpolation='bilinear'),
                        video_transforms.CenterCrop(size=(crop_size, crop_size)),
                        volume_transforms.ClipToTensor(),
                    ])
                    clip_input = transform_fn_obj(frames)
                    tick = num_frames / float(num_samples)
                    offsets = np.array([
                        int(tick / 2.0 + tick * x) for x in range(num_samples)
                    ])

                    threshold = 0.70
                    stop_id = 0
                    frame_features = []
                    frame_classes = []
                    for i_frm in offsets:
                        # print(clip_input.shape)
                        input_batch = torch.unsqueeze(
                            clip_input[:, i_frm, :, :], dim=0
                        )  # create a mini-batch as expected by the model
                        # print(input_batch.shape)
                        feat_global = []
                        try:
                            # print("input batch: ", input_batch.squeeze().shape)
                            pred, feat = expert_models['object']([input_batch.cuda().squeeze()])
                            # clip_input1 = transform_fn(frames)

                            # print(object_img.shape)
                            # import pdb; pdb.set_trace()

                            save_output = 1
                            if save_output==1:
                                rect_th = 1
                                text_size = 0.4
                                text_th = 1
                                # from PIL import Image
                                import cv2
                                transform_fn_save_img = video_transforms.Compose([
                                    video_transforms.Resize(short_side_size),
                                    video_transforms.CenterCrop(size=(crop_size, crop_size)),
                                    # volume_transforms.ClipToTensor(),
                                ])
                                clip_input1 = transform_fn_save_img(frames)
                                center_image = clip_input1[ i_frm] 
                                object_image = clip_input1[ i_frm].copy() 
                                # import pdb; pdb.set_trace()
                                # cv2.imwrite('./debug_results/pic{}_{}.png'.format(clip_name,i_frm), img)
                            
                            pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].detach().cpu().numpy())] # Get the Prediction Score
                            # print("Pred classes: ", pred_class)
                            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
                            pred_score = list(pred[0]['scores'].detach().cpu().numpy())
                            # print('Original number of bboxes: %d' % (len(pred_boxes)))

                            # Get list of index with score greater than threshold.
                            pred_threshold = [pred_score.index(x) for x in pred_score if x > threshold][-1]
                            pred_boxes_threshold = pred_boxes[:pred_threshold+1]
                            pred_class_threshold = pred_class[:pred_threshold+1]
                            # print(pred_score[:pred_threshold+1])
                            # print(pred_class_threshold)
                            # print('Number of bboxes after thresholding: %d' % (len(pred_boxes_threshold)))


                            # feature extraction
                            # print('Start feature extraction')
                            bbox_matrix = []
                            for each_box in pred_boxes_threshold:
                                bbox_matrix.append([each_box[0][0], each_box[0][1], each_box[1][0], each_box[1][1]])
                            bbox_tensor = torch.from_numpy(np.array(bbox_matrix))

                            from torchvision.ops import MultiScaleRoIAlign
                            image_shapes = (input_batch.shape[2], input_batch.shape[3])
                            box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                            output_size=7, sampling_ratio=2)
                            # The output box_features will have a dimension of 19x256x7x7
                            # 19 means number of predicted bboxes
                            # 256 is the dimension of the original feature map
                            # 7x7 is defined in MultiScaleRoIAlign's output_size.
                            feat['0'] = feat['0'].detach().cpu()
                            feat['1'] = feat['1'].detach().cpu()
                            feat['2'] = feat['2'].detach().cpu()
                            feat['3'] = feat['3'].detach().cpu()
                            box_features = box_roi_pool(feat, [bbox_tensor], [image_shapes])

                            save_output = 1
                            if save_output == 1:

                                # img = clip_input[:, i_frm, :, :].numpy()[0]


                                for i in range(len(pred_boxes_threshold)):
                                    # Draw Rectangle with the coordinates
                                    cv2.rectangle(object_image, pred_boxes_threshold[i][0], pred_boxes_threshold[i][1],
                                                color=(0, 255, 0), thickness=rect_th)
                                    # Write the prediction class
                                    print(pred_class_threshold[i])
                                    cv2.putText(object_image, pred_class_threshold[i], pred_boxes_threshold[i][0],
                                                cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness=text_th)

                                # cv2.imwrite('./debug_results/pic{}_{}.png'.format(clip_name,i_frm), img)
                                # print('Result saved.')

                            # print(box_features.shape)
                            # feat_global = torch.mean(avg_pool2d(box_features), 0).view(1,-1)
                            # clip_features.append(feat_global)
                            feat_frame = avg_pool2d(box_features).view(-1, 256)
                            feat_global = torch.mean(feat_frame, 0).reshape(1, -1)
                            # print(feat_global.shape)
                            object_features.append(feat_global)
                            frame_features = []
                            frame_classes = []
                            for i in range(len(feat_frame)):
                                frame_classes.append(pred_class_threshold[i])
                                frame_features.append(feat_frame[i])

                            clip_objects.append(frame_features)
                            # print(frame_classes)
                            start_id = stop_id
                            stop_id = len(clip_objects)
                            num_features = stop_id - start_id
                            frame_info = {
                                "classes": frame_classes,
                                "start_id": start_id,
                                "stop_id": stop_id,
                                "frame_id": i_frm,
                                "num_objects": len(frame_features)
                            }
                            # print(frame_info)
                            clip_object_class.append(frame_info)
                            # print(clip_object_class)
                            # print("--"*10)

                        except:
                            continue

                    if len(object_features) == 0:
                        object_features.append(torch.zeros(1,256).reshape(1, -1))
                    if len(clip_object_class) == 0:
                        frame_classes = 'None'
                        frame_features = torch.zeros(1,256).reshape(1, -1)
                        clip_objects.append(frame_features)
                        # print(frame_classes)
                        start_id = 0
                        stop_id = 1
                        num_features = 1
                        frame_info = {
                            "classes": frame_classes,
                            "start_id": start_id,
                            "stop_id": stop_id,
                            "frame_id": 1,
                            "num_objects": len(frame_features)
                        }
                        # print(frame_info)
                        clip_object_class.append(frame_info)
                        
              
                ### =============================================
                ### =============================================
                # print(clip_object_class)
                # print("=="20)
                #video = self.transforms(frames)
                if "howto100m" in args.modality_list:
                    #clip_input = transform_fn(frames)
                    clip_features = clip_sliding(frames,
                                                 expert_models['howto100m'])

                if "howto100m_finetune" in args.modality_list:
                    # clip_input = transform_fn(frames)
                    clip_features = clip_sliding(
                        frames, expert_models['howto100m_finetune'])


                # print(len(clip_features))
                # results_collector.extend(clip_features)
                if 'object' in args.modality_list:
                    group_objects.append(clip_objects)
                    group_object_class.append(clip_object_class)

                #print(len(results_collector))
                # tmp_feat = torch.cat(results_collector, dim=0)

                start_frame = stop_frame
                # stop_frame = len(tmp_feat)
                num_features = stop_frame - start_frame
                # print(start_frame, stop_frame, num_features, len(tmp_feat), len(results_collector), num_frames, len(video_decord))
                # center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2BGR)
                # object_image = cv2.cvtColor(object_image, cv2.COLOR_RGB2BGR)
                segment_info = {
                    "narration": clip_caption,
                    "feat_scene":scene_features,
                    "feat_appearance":appearance_features,
                    "feat_action":action_features,
                    "feat_object":object_features,
                    "image": center_image,
                    "object":object_image,
                    "segment_name": clip_name,
                    "num_frames": num_features
                }
                cv2.imwrite('./debug_results/pic{}_{}.png'.format(1,1), center_image)
                cv2.imwrite('./debug_results/pic{}_{}.png'.format(1,2), object_image)
                # import pdb; pdb.set_trace()
                video_segments.append(segment_info)

                clip_counter += 1

            # results = torch.cat(results_collector, dim=0)
            # print(len(results),"===>>>")
            # Group level information
            meta_video = {
                # "num_frames": len(results),
                "data_id": str(vid_gp_id),
                "split": split,
                "segments": video_segments
            }

            # Group features - write to h5
            # vid_h5[str(vid_gp_id)] = results.detach().cpu().numpy()
            # if 'object' in args.modality_list:
            #     gp_obj_folder = os.path.join(data_dir, 'object')
            #     os.makedirs(gp_obj_folder, exist_ok=True)

            #     gp_obj_file = os.path.join(gp_obj_folder, str(vid_gp_id) + '_obj.pickle')
            #     with open(gp_obj_file, 'wb') as f:
            #         group_obj_all = {}
            #         group_obj_all['features'] = group_objects
            #         group_obj_all['classes'] = group_object_class
            #         pk.dump(group_obj_all, f, pk.HIGHEST_PROTOCOL)


            # feat_len = int(results.shape[0])
            # del results
            # del frames_collector
            # del results_collector
            # del frames
            # del video_decord
        # else:
        #     feat_len = int(vid_h5[str(vid_gp_id)].shape[0])

        # write new meta
        meta_data[str(vid_gp_id)] = meta_video

        # gp_obj_folder = os.path.join(data_dir, 'object')
        # os.makedirs(data_dir, exist_ok=True)
        gp_obj_file = os.path.join(data_dir, 'test1k_data_vis_bgr.pickle')
        with open(gp_obj_file, 'wb') as f:
            # group_obj_all = {}
            # group_obj_all["data"] = meta_data
            # group_obj_all['features'] = group_objects
            # group_obj_all['classes'] = group_object_class
            pk.dump(meta_data, f, pk.HIGHEST_PROTOCOL)

        pbar.update()
    # vid_h5.close()
    pbar.close()

    # write new meta

    # meta_data_path = Path(
    #     os.path.join(
    #         args.data_save, "meta",
    #         "meta_group{}_{}.json".format(args.group_k, modality_name)))

    # with open(meta_data_path, "wt") as fh:
    #     write_json(meta_data, fh)

    print("=====Done!=====")


if __name__ == '__main__':
    main()
