LANGUAGE=$1
DATA_PATH=open_kbc_data/simkgc_${LANGUAGE}

python mokbc_data.py ${DATA_PATH}/

python convert_kbc.py --kge_output ${DATA_PATH}/test_data.pkl --kge_data_dir ${DATA_PATH}/ --output_dir ${DATA_PATH}/ --kbe_data_dir ${DATA_PATH}/ --output_file test_data.txt --model simkgc_${LANGUAGE} --entity_map --relation_map --filter --predictions --scores
python convert_kbc.py --kge_output ${DATA_PATH}/validation_data.pkl --kge_data_dir ${DATA_PATH}/ --output_dir ${DATA_PATH}/ --kbe_data_dir ${DATA_PATH}/ --output_file validation_data.txt --filter_val --predictions --scores_val --model simkgc_${LANGUAGE}

python convert_kbc.py --kge_output ${DATA_PATH}/train_data.pkl --kge_data_dir ${DATA_PATH}/ --output_dir ${DATA_PATH}/ --kbe_data_dir ${DATA_PATH}/ --output_file train_data.txt --predictions --model simkgc_${LANGUAGE}

# python tokenize_pkl.py --inp ${DATA_PATH}/mapped_to_ids/entity_id_map.txt --model_str bert-base-multilingual-cased
# python tokenize_pkl.py --inp ${DATA_PATH}/mapped_to_ids/relation_id_map.txt --model_str bert-base-multilingual-cased
python tokenize_pkl.py --inp ${DATA_PATH}/mapped_to_ids/entity_id_map.txt --model_str bert-base-cased
python tokenize_pkl.py --inp ${DATA_PATH}/mapped_to_ids/relation_id_map.txt --model_str bert-base-cased
