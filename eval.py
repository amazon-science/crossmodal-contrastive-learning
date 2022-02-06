import argparse
from multiprocessing import cpu_count
import os
from pathlib import Path

from data_loader.videotext_dataloader import create_datasets, create_loaders
from trainer.trainer import TrainerVideoText
from utils_collection.general import load_config, set_seed, print_csv_results
from utils_collection.general import create_dataloader_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=str, help='Experiment to run')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
    parser.add_argument('--group_k',
                        type=int,
                        default=5,
                        help='number of segments per video')
    parser.add_argument("--modality",
        type=str,
        default="howto100m",
        help="select the modality name")
    parser.add_argument('--workers',
                        type=int,
                        default=None,
                        help='set number of workers (default #CPUs - 1)')
    parser.add_argument('--log_dir',
                        type=str,
                        default="runs/eval",
                        help='directory to save/load the runs and logs')
    parser.add_argument("--val_split",
                        type=str,
                        default="test1k",
                        help="Select the validation split")
    parser.add_argument("--dataroot",
                        type=str,
                        default="data",
                        help="change datasets root path")
    parser.add_argument("--cuda", action="store_true", help="train on GPUs")
    parser.add_argument("--single_gpu",
                        action="store_true",
                        help="Disable multi-GPU")
    parser.add_argument("--preload_vid",
                        action="store_true",
                        help="Load video features into RAM")
    args = parser.parse_args()


    configuration = load_config(args.config)
    set_seed(0)
    num_workers = min(10,
                      cpu_count() -
                      1) if args.workers is None else args.workers

    # meta_data_path = Path(os.path.join(args.dataroot, "meta", "meta_group{}.json".format(args.group_k)))
    # video_feat_path = Path(os.path.join(args.dataroot, "group{}".format(args.group_k), "video_features", "howto_h100m.h5"))
    # language_feat_path = Path(os.path.join(args.dataroot, "group{}".format(args.group_k), "language_features", "text_default.h5"))   
    # meta_text_len_path = Path(os.path.join(args.dataroot, "group{}".format(args.group_k), "language_features", "text_lens_default.json"))

    # data_path_dic = {
    #     "meta_data": meta_data_path,
    #     "video_feats": video_feat_path,
    #     "language_feats": language_feat_path,
    #     "meta_text_len": meta_text_len_path,
    #     "dataset_name": configuration.dataset.name
    # }

    data_path_dict = create_dataloader_path(args.dataroot, args.group_k, configuration.dataset.name, video_feature_name=args.modality)

    configuration.dataset.val_split = args.val_split
    val_set = create_datasets(data_path_dict, configuration, args.preload_vid, True, eval=True)
    val_loader = create_loaders([], val_set,
                                              configuration.training.batch_size,
                                              num_workers, eval=True)

    trainer = TrainerVideoText(args.log_dir, configuration, args.cuda, args.cuda
                               and not args.single_gpu, args.checkpoint, False)
    trainer.validate(val_loader)
    trainer.close()



if __name__ == '__main__':
    main()
