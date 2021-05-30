# CEAR
Pretrained language models like BERT have shown to store factual knowledge about the world. This knowledge can be used to augment Knowledge Bases, which are often incomplete. However, prior attempts at using BERT for task of Knowledge Base Completion (KBC) resulted in performance worse than the embedding based techniques that only use the graph structure. In this work we develop a novel model, Cross-Entity Aware Reranker (CEAR), that uses BERT to re-rank the output of existing KBC models. Unlike prior works that score each entity independently, CEAR jointly scores the top–k entities from embedding based KBC models, using cross-entity attention in BERT. CEAR establishes a new state of the art performance with 42.6 HITS@1 in FB15k-237 (32.7% relative improvement) and 5.3 pt improvement in HITS@1 for Open Link Prediction. 

# References
Code in KnowledgeGraphEmbeddings has been taken from https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding

Code in extremeText has been taken from https://github.com/mwydmuch/extremeText

# Installing pre-reqs
```
conda create --name cear python=3.6
conda activate cear
cd cear
pip install -r requirements.txt
```

# Downloading the OLPBENCH dataset
```
wget http://data.dws.informatik.uni-mannheim.de/olpbench/olpbench.tar.gz
tar xzf olpbench.tar.gz
```

## Contents
./mapped_to_ids
./mapped_to_ids/entity_id_map.txt
./mapped_to_ids/entity_token_id_map.txt
./mapped_to_ids/relation_id_map.txt
./mapped_to_ids/relation_token_id_map.txt
./mapped_to_ids/entity_id_tokens_ids_map.txt
./mapped_to_ids/relation_id_tokens_ids_map.txt
./mapped_to_ids/validation_data_linked_mention.txt
./mapped_to_ids/validation_data_linked.txt
./mapped_to_ids/validation_data_all.txt
./mapped_to_ids/test_data.txt
./mapped_to_ids/train_data_thorough.txt
./mapped_to_ids/train_data_basic.txt
./mapped_to_ids/train_data_simple.txt

./train_data_simple.txt
./train_data_basic.txt
./train_data_thorough.txt

./validation_data_all.txt
./validation_data_linked_mention.txt
./validation_data_linked.txt

./test_data.txt

./README.md

## Format

### train, validation and test files

The train, validation and test files contain triples as described in the publication. The format is 5 TAB separated columns:

COL 1			COL 2			COL 3			COL 4					COL 5
subject mention tokens	open relation tokens	object mention tokens	alternative subj mention|||...|||...	alternative obj mention|||...|||...

Except for validation_data_linked.txt test_data.txt COL 4 and COL 5 are empty.


### mapped_to_ids

mapped_to_ids contains the training, validation and test data mapped to ids according to the maps:

entity_id_map.txt:		maps entities (actually entity mentions) to ids, starting from 2 (!)
relation_id_map.txt:		maps relations to ids, starting from 2 (!)

entity_token_id_map.txt:	maps the tokens of entities to ids, starting from 4 (!)
relation_token_id_map.txt:	maps the tokens of relations to ids, starting from 4 (!)

entity_id_tokens_ids_map.txt:	maps entity ids to a sequence of token ids
relation_id_tokens_ids_map.txt:	maps relation ids to a sequence of token ids

The train, validation and test files contain triples as described in the publication. The format is 5 TAB separated columns, COL 4 and COL 5 are lists of space seperated ids:

COL 1		COL 2           COL 3		COL 4			COL 5
entity id	relation id	entity id	alt. subj entity ids	alt. obj entity ids


# Open Link Prediction (OLPBENCH)
This section contains instruction for training the extreme text model (original github repo: https://github.com/mwydmuch/extremeText) and then training BERT over the predictions of this extreme text model.   
To avoid confusion, each step begins from the root of the repository.
## Install extremetext
```
cd extremeText
make
```
## Open KBs 
### Step 1: convert open kb data to xt format
```
python3 helper_scripts/sample_1mil_train.py
python3 helper_scripts/get_freq_for-relation.py
python3 helper_scripts/generate_all_knowns.py
cd extremeText/data
ln -s ../../olpbench olpbench
python xt_input.py --inp olpbench/test_data.txt --type head --num_frequent 5
python xt_input.py --inp olpbench/test_data.txt --type tail --num_frequent 5
python xt_input.py --inp olpbench/train_data_thorough.txt --type head --num_frequent 5
python xt_input.py --inp olpbench/train_data_thorough.txt --type tail --num_frequent 5
python xt_input.py --inp olpbench/train_data_thorough_1mil.txt --type head --num_frequent 5
python xt_input.py --inp olpbench/train_data_thorough_1mil.txt --type tail --num_frequent 5
python xt_input.py --inp olpbench/validation_data_linked.txt --type head --num_frequent 5
python xt_input.py --inp olpbench/validation_data_linked.txt --type tail --num_frequent 5
```

### Step 2: Train stage 1 extreme text model
```
cd extremeText
mkdir data/olpbench/xt_models
./extremetext supervised -input data/olpbench/train_data_thorough.txt.tail.xt -output data/olpbench/xt_models/tail_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2
./extremetext supervised -input data/olpbench/train_data_thorough.txt.head.xt -output data/olpbench/xt_models/head_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2
```

### Step 3: Get stage 1 predictions and convert data for stage 2 
Test data:
```
MODEL=thorough_f5_d300
FILE=olpbench/test_data.txt
THREADS=2

cd extremeText
cd data
../extremetext predict-prob olpbench/xt_models/head_$MODEL.bin $FILE.head.xtp 50 0 $FILE.head_$MODEL'.preds' $THREADS
cat $FILE.head_$MODEL'.preds'.0* > $FILE.head_$MODEL'.preds_parallel'
rm $FILE.head_$MODEL'.preds'.0*
mv $FILE.head_$MODEL'.preds_parallel' $FILE.head_$MODEL'.preds'

../extremetext predict-prob olpbench/xt_models/tail_$MODEL.bin $FILE.tail.xtp 50 0 $FILE.tail_$MODEL'.preds' $THREADS
cat $FILE.tail_$MODEL'.preds'.0* > $FILE.tail_$MODEL'.preds_parallel'
rm $FILE.tail_$MODEL'.preds'.0*
mv $FILE.tail_$MODEL'.preds_parallel' $FILE.tail_$MODEL'.preds'

python xt_output.py --inp $FILE --model $MODEL --type head
python xt_output.py --inp $FILE --model $MODEL --type tail
```

Validation data:
```
MODEL=thorough_f5_d300
FILE=olpbench/validation_data_linked.txt
THREADS=2

cd extremeText
cd data
../extremetext predict-prob olpbench/xt_models/head_$MODEL.bin $FILE.head.xtp 50 0 $FILE.head_$MODEL'.preds' $THREADS
cat $FILE.head_$MODEL'.preds'.0* > $FILE.head_$MODEL'.preds_parallel'
rm $FILE.head_$MODEL'.preds'.0*
mv $FILE.head_$MODEL'.preds_parallel' $FILE.head_$MODEL'.preds'

../extremetext predict-prob olpbench/xt_models/tail_$MODEL.bin $FILE.tail.xtp 50 0 $FILE.tail_$MODEL'.preds' $THREADS
cat $FILE.tail_$MODEL'.preds'.0* > $FILE.tail_$MODEL'.preds_parallel'
rm $FILE.tail_$MODEL'.preds'.0*
mv $FILE.tail_$MODEL'.preds_parallel' $FILE.tail_$MODEL'.preds'

python xt_output.py --inp $FILE --model $MODEL --type head
python xt_output.py --inp $FILE --model $MODEL --type tail
```

Train data:
We will only generate predictions for 1 million train points (for training stage 2 in reasonable time)
```
MODEL=thorough_f5_d300
FILE=olpbench/train_data_thorough_1mil.txt
THREADS=60

cd extremeText
cd data
../extremetext predict-prob olpbench/xt_models/head_$MODEL.bin $FILE.head.xtp 50 0 $FILE.head_$MODEL'.preds' $THREADS
cat $FILE.head_$MODEL'.preds'.0* > $FILE.head_$MODEL'.preds_parallel'
rm $FILE.head_$MODEL'.preds'.0*
mv $FILE.head_$MODEL'.preds_parallel' $FILE.head_$MODEL'.preds'

../extremetext predict-prob olpbench/xt_models/tail_$MODEL.bin $FILE.tail.xtp 50 0 $FILE.tail_$MODEL'.preds' $THREADS
cat $FILE.tail_$MODEL'.preds'.0* > $FILE.tail_$MODEL'.preds_parallel'
rm $FILE.tail_$MODEL'.preds'.0*
mv $FILE.tail_$MODEL'.preds_parallel' $FILE.tail_$MODEL'.preds'

python xt_output.py --inp $FILE --model $MODEL --type head
python xt_output.py --inp $FILE --model $MODEL --type tail
```

Move all this data to a separate folder
```
cd cear
mkdir open_kbc_data
mv ../olpbench/*stage1 open_kbc_data/ 
cd open_kbc_data
ln -s ../../olpbench/mapped_to_ids mapped_to_ids
cp ../../olpbench/all_knowns.pkl ./
cd ../
python3 tokenize_pkl.py --inp open_kbc_data/mapped_to_ids/relation_id_map.txt --model_str bert-base-uncased
python3 tokenize_pkl.py --inp open_kbc_data/mapped_to_ids/entity_id_map.txt --model_str bert-base-uncased
```

### Step 4: Train stage 2 model
```
cd cear
python run.py --save open_kbc_data/models --mode train --gpus 1 --epochs 5 --stage2 --negative_samples 30 --data_dir open_kbc_data --model mcq --stage1_model thorough_f5_d300 --model_str bert-base-uncased --task_type both --max_tokens 5000 --train open_kbc_data/train_data_thorough_1mil.txt
```

### Step 5: Test stage 1 predictions
```
cd cear
python run.py --save /tmp --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 30 --data_dir open_kbc_data --model mcq --stage1_model thorough_f5_d300 --model_str bert-base-uncased --task_type both --xt_results --test open_kbc_data/test_data.txt
```
Expected results: H@1: 6.4, H@10: 16.3, H@50: 26.0

### Step 6: Test stage 2 predictions
```
cd cear
python run.py --save /tmp --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 30 --data_dir open_kbc_data --model mcq --stage1_model thorough_f5_d300 --model_str bert-base-uncased --task_type both --checkpoint <path to model checkpoint> --test open_kbc_data/test_data.txt
```
Expected results: H@1: 7.4, H@10: 17.9, H@50: 26.0

# Closed Link Prediction (FB15K-237, WN18-RR)
This section contains instructions to train the stage 1 model - `CompleX` and `RotatE` - on different datasets - `FB15K237` and `WN18RR` - and then using the predictions from this model, we can train the stage 2 model.  
To avoid confusion, each step begins from the root of the repository.
### Step 1: Train stage 1 model
```
cd KnowledgeGraphEmbedding
mkdir models
```
Select the appropriate command for the right combination of dataset and model:

ComplEx on FB15K237
```
bash run.sh train ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
```
ComplEx on WN18RR
```
bash run.sh train ComplEx WN18RR 0 0 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005
```
RotatE on FB15K237
```
bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de
```
RotatE on WN18RR
```
bash run.sh train RotatE WN18RR 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de
```

### Step 2: Generate predictions from this stage 1 model
```
cd KnowledgeGraphEmbedding
```
Select the appropriate command for the right combination of dataset and model:
ComplEx on FB15K237
```
bash run.sh predict_train ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash run.sh predict_test ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash run.sh predict_val ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
```
ComplEx on WN18RR
```
bash run.sh predict_train ComplEx WN18RR 0 0 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005
bash run.sh predict_test ComplEx WN18RR 0 0 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005
bash run.sh predict_val ComplEx WN18RR 0 0 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005
```
RotatE on FB15K237
```
bash run.sh predict_train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de
bash run.sh predict_test RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de
bash run.sh predict_val RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de
```
RotatE on WN18RR
```
bash run.sh predict_train RotatE WN18RR 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de
bash run.sh predict_test RotatE WN18RR 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de
bash run.sh predict_val RotatE WN18RR 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de
```
This will generate `output_train.pkl`, `output_test.pkl` and `output_valid.pkl` inside the appropriate model folder. We will move this data to right place and convert it into a format that is acceptable by our 2 stage architecture.
```
export DATASET=FB15k-237 or WN18RR
export MODEL=RotatE or ComplEx
cd ../cear
mkdir closed_kbc_data/$MODEL
mkdir closed_kbc_data/$MODEL/data
mkdir closed_kbc_data/$MODEL/data/$DATASET
cp ../KnowledgeGraphEmbedding/data/$DATASET/*dict closed_kbc_data/$MODEL/data/$DATASET
mv ../KnowledgeGraphEmbedding/models/$MODEL"_"$DATASET"_"0/output_train.pkl closed_kbc_data/$MODEL/data/$DATASET/train_data.pkl
mv ../KnowledgeGraphEmbedding/models/$MODEL"_"$DATASET"_"0/output_test.pkl closed_kbc_data/$MODEL/data/$DATASET/test_data.pkl
mv ../KnowledgeGraphEmbedding/models/$MODEL"_"$DATASET"_"0/output_valid.pkl closed_kbc_data/$MODEL/data/$DATASET/validation_data.pkl
python convert_kbc.py --kge_output closed_kbc_data/$MODEL/data/$DATASET/test_data.pkl --kbe_data_dir closed_kbc_data/kg-bert --kge_data_dir closed_kbc_data/$MODEL/ --dataset $DATASET --output_dir closed_kbc_data/data_for_stage2/$DATASET --output_file test_data.txt --model $MODEL --entity_map --relation_map --filter --predictions --scores
python convert_kbc.py --kge_output closed_kbc_data/$MODEL/data/$DATASET/validation_data.pkl --kbe_data_dir closed_kbc_data/kg-bert --kge_data_dir closed_kbc_data/$MODEL/ --dataset $DATASET --output_dir closed_kbc_data/data_for_stage2/$DATASET --output_file validation_data.txt --model $MODEL --filter_val --predictions --scores_val
python convert_kbc.py --kge_output closed_kbc_data/$MODEL/data/$DATASET/train_data.pkl --kbe_data_dir closed_kbc_data/kg-bert --kge_data_dir closed_kbc_data/$MODEL/ --dataset $DATASET --output_dir closed_kbc_data/data_for_stage2/$DATASET --output_file train_data.txt --model $MODEL --predictions
python3 tokenize_pkl.py --inp closed_kbc_data/data_for_stage2/$DATASET/mapped_to_ids/entity_id_map.txt --model_str bert-base-cased
python3 tokenize_pkl.py --inp closed_kbc_data/data_for_stage2/$DATASET/mapped_to_ids/relation_id_map.txt --model_str bert-base-cased
```
The data is ready for 2 stage architecture

### Step 3: Train stage 2 model
```
export DATASET=FB15k-237 or WN18RR
export MODEL=RotatE or ComplEx
cd cear
python run.py --save closed_kbc_data/models/$MODEL"_"$DATASET --mode train --gpus 1 --epochs 10 --stage2 --negative_samples 40 --data_dir closed_kbc_data/data_for_stage2/$DATASET --model mcq --stage1_model $MODEL --model_str bert-base-cased --task_type both --max_tokens 5000 --ckbc --limit_tokens 10
```
This will save a model inside `closed_kbc_data/models/$MODEL_$DATASET`

### Step 4: Test stage 1 predictions
```
export DATASET=FB15k-237 or WN18RR
export MODEL=RotatE or ComplEx
cd cear
python run.py --save /tmp --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 40 --data_dir closed_kbc_data/data_for_stage2/$DATASET --model mcq --stage1_model $MODEL --model_str bert-base-cased --task_type both --ckbc --xt_results --test closed_kbc_data/data_for_stage2/$DATASET/test_data.txt --limit_tokens 10
```
Expected results:
1. ComplEx FB15K237: MRR: 0.32, H@1: 23, H@10: 51.3
2. ComplEx WN18RR: MRR: 0.47, H@1: 42.8, H@10: 55.5
3. RotatE FB15K237: MRR: 0.34, H@1: 23.8, H@10: 53.1
4. RotatE WN18RR: MRR: 0.47, H@1: 42.3, H@10: 57.3

### Step 5: Test stage 2 predictions
```
export DATASET=FB15k-237 or WN18RR
export MODEL=RotatE or ComplEx
cd cear
python run.py --save /tmp --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 40 --data_dir closed_kbc_data/data_for_stage2/$DATASET --model mcq --stage1_model $MODEL --model_str bert-base-cased --task_type both --ckbc --checkpoint <path to model checkpoint> --test closed_kbc_data/data_for_stage2/$DATASET/test_data.txt --limit_tokens 10
```
Expected results:
1. ComplEx FB15K237: MRR: 0.48, H@1: 42.2, H@10: 57.9  
2. ComplEx WN18RR: MRR: 0.47, H@1: 43 H@10: 54.3
3. RotatE FB15K237: MRR: 0.45, H@1: 38.3, H@10: 56.7
4. RotatE WN18RR: MRR: 0.49, H@1: 44.3, H@10: 56.5

## Important config variables
1. `negative_samples` in `cear/run.py` can be used to change the number of stage 1 samples used.
2. `task_type` in `cear/run.py` can be `tail|head|both` for approriate training or testing (head link prediction / tail link prediction).
3. `from_scratch` can be used with `cear/run.py` to train a model without pretrained-bert knowledge.
4. 'shuffle' can be used with `cear/run.py` to shuffle the stage 1 samples before training the stage 2 model.
5. `model` can be set to `lm` instead of `mcq` to train a model without cross entity attention of stage 1 samples. (In this case each sample would be fed independently to bert model)


## Extend stage 2 to any architecture
This architecture can be used on top of any model on any dataset. To do this the stage 1 model must have yielded the following files:(Refer to files created above in Closed Link Prediction for better understanding)
1. entities.dict - contains id for each entity used by stage 1 model
2. relations.dict - contains id for each relation used by stage 1 model
3. output_train.pkl - a dictionary of the form {
	(entity1_id, relation_id, entity2_id): {
		"head-batch":{
			"index": a list of ids of best k entities for head prediction, 
			"confidence": a list of condifence scores for above k predictions, 
			"bias": a list of known entities for (entity1_id, relation_id) for filtered evaluation,
			"score":{
					"MRR": mean filtered rank of entity2_id for stage1 model, 
					"HITS1": mean filtered hits1 of entity2_id for stage1 model,
					"HITS3": mean filtered hits3 of entity2_id for stage1 model,
					"HITS10": mean filtered hits10 of entity2_id for stage1 model,
				}
		},
		"tail-batch":{
			"index": a list of ids of best k entities for tail prediction, 
			"confidence": a list of condifence scores for above k predictions, 
			"bias": a list of known entities for (entity2_id, relation_id) for filtered evaluation,
			"score":{
					"MRR": mean filtered rank of entity1_id for stage1 model, 
					"HITS1": mean filtered hits1 of entity1_id for stage1 model,
					"HITS3": mean filtered hits3 of entity1_id for stage1 model,
					"HITS10": mean filtered hits10 of entity1_id for stage1 model,
				}
		} 
	},...
}
4. output_test.pkl
5. output_val.pkl
Next use `cear/convert_kbc.py` as done in previous section with similar folder structures to convert this into required format for stage 2 training.





