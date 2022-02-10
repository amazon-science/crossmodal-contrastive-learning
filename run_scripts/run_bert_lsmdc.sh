#run bert model to extract features and create json file
python3 -m utils_collection.run_bert --dataroot /mnt/efs/fs1/workspace/experiments/data/lsmdc16/mmt_experts/debug2/test --group_k 1 --cuda --modality mmt
# python3 -m utils_collection.run_bert --dataroot /mnt/efs/fs1/workspace/experiments/data/lsmdc16/modality_experts/faces --group_k 5 --modality face --cuda
