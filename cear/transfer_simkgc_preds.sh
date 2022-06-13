LANGUAGE=$1
SIMKGC_PATH=/home/ee/btech/ee1180957/scratch/BTP/SimKGC/checkpoint/mopenkb_${LANGUAGE}
CEAR_PATH=open_kbc_data/simkgc_${LANGUAGE}

# SIMKGC_PATH=/home/ee/btech/ee1180957/scratch/BTP/SimKGC/checkpoint/ReVerb45K
# CEAR_PATH=open_kbc_data/simkgc_ReVerb45K

mkdir ${CEAR_PATH}

cp ${SIMKGC_PATH}/eval_test.txt.json_forward_model_best.mdl.json ${CEAR_PATH}/test_forward.json
cp ${SIMKGC_PATH}/eval_test.txt.json_backward_model_best.mdl.json ${CEAR_PATH}/test_backward.json

cp ${SIMKGC_PATH}/eval_valid.txt.json_forward_model_best.mdl.json ${CEAR_PATH}/val_forward.json
cp ${SIMKGC_PATH}/eval_valid.txt.json_backward_model_best.mdl.json ${CEAR_PATH}/val_backward.json

cp ${SIMKGC_PATH}/eval_train.txt.json_forward_model_best.mdl.json ${CEAR_PATH}/train_forward.json
cp ${SIMKGC_PATH}/eval_train.txt.json_backward_model_best.mdl.json ${CEAR_PATH}/train_backward.json