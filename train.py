" Run MultiModalExpert model for video-language representation learning"

import argparse
from multiprocessing import cpu_count
import os
from pathlib import Path
from timeit import default_timer as timer

from data_loader.videotext_dataloader import create_datasets, create_loaders
from trainer.trainer import TrainerVideoText
from utils_collection.general import load_config, set_seed, print_csv_results
from utils_collection.general import create_dataloader_path
import utils_collection.arguments as arguments
from utils_collection.yaml_config import load_yaml_config_file
from utils_collection.config_tools import RetrievalConfig as Config

EXP_TYPE = "retrieval"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments.add_default_args(parser)  # logging level etc.
    arguments.add_exp_identifier_args(parser)  # arguments to identify the experiment to run
    arguments.add_trainer_args(parser)  # general trainer arguments
    arguments.add_dataset_test_arg(parser)  # flag for dataset testing

    parser.add_argument("--modality_list", nargs="+", default=["None"],
        help="select the first modality name (can be more than one modality)")
    parser.add_argument("--second_modality", default="None",
        help="select the second modality name (only one modality)")
    parser.add_argument('--group_k',
                        type=int,
                        default=5,
                        help='number of segments per video')
    parser.add_argument("--dataroot",
                        type=str,
                        default="data",
                        help="dataset path")
    parser.add_argument("--data_pickle",
                        type=str,
                        default="",
                        help="Set to pickle folder path")
    parser.add_argument("--val_data",
                        type=str,
                        default="",
                        help="change datasets val path")
    parser.add_argument("--test_data",
                    type=str,
                    default="test1k",
                    help="Set test data path")
    parser.add_argument("--cuda", action="store_true", help="train on GPUs")
    parser.add_argument("--eval", action="store_true", help="Evaluate on test set")
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
    parser.add_argument('--save_path', type=str, help='Path to save embeddings')

    args = parser.parse_args()
    # configuration = load_config(args.config)
    # load repository config yaml file to dict
    exp_group, exp_name, config_file = arguments.setup_experiment_identifier_from_args(args, EXP_TYPE)
    config = load_yaml_config_file(config_file)

    # update experiment config and dataset path given the script arguments
    # path_data = arguments.update_path_from_args(args)

    configuration = arguments.update_config_from_args(config, args)
        # read experiment config dict
    configuration = Config(configuration, is_train=not args.validate and not args.test_dataset)
    if args.print_config:
        print(configuration)

    # create dataset and dataloader
    # if configuration.dataset.preload_data:
    #     cmd = "ulimit -n 2000000"
    #     print(f"Run system command to avoid TooManyFiles error:\n{cmd}")
    #     os.system(cmd)

    if configuration.random_seed is not None:
        print(f"Set seed to {configuration.random_seed}")
        set_seed(configuration.random_seed, set_deterministic=False)


    global_timer = timer() # global timer
    
    if args.modality_list[0] == 'None':
        feature_name_list_a = configuration.dataset.modality_feat_name_a
    else:
        feature_name_list_a = args.modality_list

    if args.second_modality == 'None':
        feature_name_b = configuration.dataset.modality_feat_name_b
    else:
        feature_name_b = args.second_modality

    data_path_dict = create_dataloader_path(args.dataroot, args.group_k,
     configuration.dataset.name, feature_name_modality_a=feature_name_list_a, feature_name_modality_b=feature_name_b,
      pickle_path=args.data_pickle)
    
    if not args.eval:
        train_set, val_set = create_datasets(data_path_dict, configuration, configuration.dataset.preload_data,
                                            configuration.dataset.preload_data, debug_train_size=configuration.train.debug_size,
                                            debug_val_size=configuration.val.debug_size, pickle_path=data_path_dict["pickle_path"])
        
        # Change the default validation set
        if args.val_data != "":
            val_path_dict = create_dataloader_path(args.val_data, 1, configuration.dataset.name,
            feature_name_modality_a=feature_name_list_a, feature_name_modality_b=feature_name_b, pickle_path=args.data_pickle)
            val_set = create_datasets(val_path_dict, configuration, configuration.dataset.preload_data, 
            configuration.dataset.preload_data, eval=True, pickle_path=data_path_dict["pickle_path"])

    # Create test set for inference
    if args.test_data != "":
        path_test = os.path.join(args.dataroot, args.test_data)
        test_path_dict = create_dataloader_path(path_test, 1, configuration.dataset.name, 
        feature_name_modality_a=feature_name_list_a, feature_name_modality_b=feature_name_b, pickle_path=args.data_pickle)
        test_set = create_datasets(test_path_dict, configuration, configuration.dataset.preload_data,
         configuration.dataset.preload_data, test=True, debug_test_size=configuration.test.debug_size, pickle_path=data_path_dict["pickle_path"])
    else: 
        test_set = None

    # Create data loaders


    if args.eval:
        test_loader = create_loaders(train_set=None, val_set=None, test_set=test_set,
                                              batch_size=configuration.train.batch_size,
                                              num_workers=configuration.dataset.num_workers, eval=True)
        run_name = f"test"
        trainer = TrainerVideoText(configuration, feature_name_list_a, feature_name_b, exp_group, exp_name, run_name, len(test_loader), log_dir=args.log_dir,
            log_level=args.log_level, logger=None, reset=args.reset, load_ckpt=args.checkpoint, save_emb_path=args.save_path)
        trainer.test(test_loader)
    else:
        train_loader, val_loader, test_loader = create_loaders(train_set, val_set, test_set,
                                              configuration.train.batch_size,
                                              configuration.dataset.num_workers)

        for run_number in range(1, args.num_runs + 1):
            run_name = f"{args.run_name}{run_number}"
            trainer = TrainerVideoText(configuration, feature_name_list_a, feature_name_b, exp_group, exp_name, run_name, len(train_loader), log_dir=args.log_dir,
                log_level=args.log_level, logger=None, reset=args.reset)

            trainer.logger.info("---------- Start training ... ----------")
            if args.val_data != "":
                trainer.logger.info("Validation data default changed to : {}".format(args.val_data))
            trainer.train_loop(train_loader, val_loader, test_loader)

            trainer.logger.info("---------- Results ----------")
            trainer.logger.info(trainer.exp.path_logs)
            # print_csv_results(trainer.log_dir / "train_metrics.csv",
            #                         configuration,
            #                         print_fn=trainer.logger.info)

        end_global_timer = timer()
        trainer.logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))

    trainer.close()
    del trainer



if __name__ == '__main__':
    main()