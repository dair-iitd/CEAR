# RotatE_WN18RR
python run.py --save closed_kbc_data/models/arr_1_submission/RotatE_WN18RR --mode test --gpus 1 --epochs 10 --stage2 --negative_samples 44 --data_dir closed_kbc_data/data_for_stage2/WN18RR --model mcq --stage1_model RotatE --model_str bert-base-cased --task_type both --max_tokens 5000 --ckbc --limit_tokens 10 --filter_train --checkpoint closed_kbc_data/models/arr_1_submission/RotatE_WN18RR/epoch=05_loss=0.026_eval_acc=0.442.ckpt

HITS@1: 44.19

# ComplEx_FB15k-237
python run.py --save closed_kbc_data/models/arr_1_submission/ComplEx_FB15k-237 --mode test --gpus 1 --epochs 10 --stage2 --negative_samples 44 --data_dir closed_kbc_data/data_for_stage2/FB15k-237 --model mcq --stage1_model ComplEx --model_str bert-base-cased --task_type both --max_tokens 5000 --ckbc --limit_tokens 10 --filter_train --checkpoint closed_kbc_data/models/arr_1_submission/ComplEx_FB15k-237/epoch=08_loss=0.043_eval_acc=0.428.ckpt

HITS@1: 42.2

python run.py --save open_kbc_data/models --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 30 --data_dir open_kbc_data --model mcq --stage1_model thorough_f5_d300 --model_str bert-base-uncased --task_type both --checkpoint open_kbc_data/models/epoch=01_loss=0.078_eval_acc=0.072.ckpt --test open_kbc_data/test_data.txt

HITS@1: 7.2
