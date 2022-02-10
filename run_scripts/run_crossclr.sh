python3 train.py --config_file config/lsmdc.yaml --cuda \
--dataroot /mnt/efs/fs1/workspace/experiments/data/lsmdc16/modality_experts/ \
--data_pickle /mnt/efs/fs1/workspace/experiments/data/lsmdc16/modality_experts/modalities_pickle_v2/ \
--group_k 5 --exp_group mod_mod_combine \
--exp_name text_to_action_appearance_B96_D2048_milNCE_r2 --num_runs 2 --test_data test1k \
--second_modality text \
--modality_list action appearance \
# --eval \ #Uncomment this line and next two lines for evaluation and extracting features! specify the model and save path.
# --save_path /mnt/efs/fs1/workspace/xray_transformer_v1/src/lablet_multimodal_transformer_xray/extract_features \
# --checkpoint /mnt/efs/fs1/workspace/xray_transformer_v1/src/lablet_multimodal_transformer_xray/experiments/retrieval/mod_mod_T3/D768_headL16H4_LR6e4_PoolMax_Drop2e2_B64_TW75_crossCLr_r1_run2/models/model_4.pth

#================= modalities =====================
# # action,scene,appearance,howto100m_finetune,object


