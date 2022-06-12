LANGUAGE='en'

# Generating the data
```
pythom mokbc_data.py open_kbc_data/simkgc_${LANGUAGE}

python convert_kbc.py --kge_output open_kbc_data/simkgc_en/test_data.pkl --kge_data_dir open_kbc_data/simkgc_en/ --output_dir open_kbc_data/simkgc_en/ --kbe_data_dir open_kbc_data/simkgc_en/ --output_file test_data.txt --model simkgc_en --entity_map --relation_map --filter --predictions --scores
python convert_kbc.py --kge_output open_kbc_data/simkgc_en/validation_data.pkl --kge_data_dir open_kbc_data/simkgc_en/ --output_dir open_kbc_data/simkgc_en/ --kbe_data_dir open_kbc_data/simkgc_en/ --output_file validation_data.txt --filter_val --predictions --scores_val --model simkgc_en

python convert_kbc.py --kge_output open_kbc_data/simkgc_en/train_data.pkl --kge_data_dir open_kbc_data/simkgc_en/ --output_dir open_kbc_data/simkgc_en/ --kbe_data_dir open_kbc_data/simkgc_en/ --output_file train_data.txt --predictions --model simkgc_en

python tokenize_pkl.py --inp open_kbc_data/simkgc_en/mapped_to_ids/entity_id_map.txt --model_str bert-base-cased
python tokenize_pkl.py --inp open_kbc_data/simkgc_en/mapped_to_ids/relation_id_map.txt --model_str bert-base-cased
```
# Training the model
```
python run.py --save open_kbc_data/models/simgkc_en/ --mode train --gpus 1 --epochs 10 --stage2 --negative_samples 10 --data_dir open_kbc_data/simkgc_en --model mcq --stage1_model simkgc_en --model_str bert-base-cased --task_type both --max_tokens 5000 --ckbc --limit_tokens 10 
```

# Testing the model
```
python run.py --save open_kbc_data/models/simkgc_en/ --mode test --gpus 1 --epochs 10 --stage2 --negative_samples 10 --data_dir open_kbc_data/simkgc_en --model mcq --stage1_model simkgc_en --model_str bert-base-cased --task_type both --max_tokens 5000 --ckbc --limit_tokens 10 --checkpoint open_kbc_data/models/simkgc_simkgc/epoch=05_loss=0.007_eval_acc=0.021.ckpt

python run.py --save open_kbc_data/models/simkgc_en/ --mode test --gpus 1 --epochs 10 --stage2 --negative_samples 10 --data_dir open_kbc_data/simkgc_en --model mcq --stage1_model simkgc_en --model_str bert-base-cased --task_type both --max_tokens 5000 --ckbc --limit_tokens 10 --checkpoint open_kbc_data/models/simkgc_simkgc/epoch=05_loss=0.007_eval_acc=0.021.ckpt --xt_results
```
