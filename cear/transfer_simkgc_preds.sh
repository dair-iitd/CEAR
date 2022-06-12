SIMKGC_PATH="/home/ee/btech/ee1180957/scratch/BTP/SimKGC/checkpoint/mopenkb_hi"
cp ${SIMKGC_PATH}/eval_test.txt.json_forward_model_best.mdl.json open_kbc_data/simkgc_hi/test_forward.json
cp ${SIMKGC_PATH}/eval_test.txt.json_backward_model_best.mdl.json open_kbc_data/simkgc_hi/test_backward.json

cp ${SIMKGC_PATH}/eval_valid.txt.json_forward_model_best.mdl.json open_kbc_data/simkgc_hi/val_forward.json
cp ${SIMKGC_PATH}/eval_valid.txt.json_backward_model_best.mdl.json open_kbc_data/simkgc_hi/val_backward.json

cp ${SIMKGC_PATH}/eval_train.txt.json_forward_model_best.mdl.json open_kbc_data/simkgc_hi/train_forward.json
cp ${SIMKGC_PATH}/eval_train.txt.json_backward_model_best.mdl.json open_kbc_data/simkgc_hi/train_backward.json