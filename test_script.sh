python test.py \
--resume './outputs/train_DuRDN/checkpoints/model_999.pt' \
--resume_epoch 999 \
--experiment_name 'test_DuRDN_999' \
--model_type 'model_cnn' \
--data_root '../../Data/Dataset_filename' \
--net_G 'DuRDN' \
--patch_size_eval 32 32 32 \
--lr 5e-4 \
--norm 'BN' \
--norm_pred_AC \
--use_scatter \
--use_scatter2 \
--use_scatter3 \
--use_gender \
--use_bmi \
--use_state \
--gpu_ids 0



