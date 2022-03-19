python train.py \
--experiment_name 'train_DuRDN' \
--model_type 'model_cnn' \
--data_root '../../Data/Dataset_filename/' \
--net_G 'DuRDN' \
--batch_size 2 \
--n_patch_train 1 \
--patch_size_train 32 32 32 \
--eval_epochs 10 \
--save_epochs 10 \
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



