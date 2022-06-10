
## Run ExtermeText for ReVerb20K and ReVerb45K
# Initial results are very bad... HITS@1 of only 4% Tail and 0.3% Head

DATA_DIR=olpbench/ReVerb20K/

python xt_input.py --data_dir $DATA_DIR --inp test_data.txt --type head --num_frequent 5
python xt_input.py --data_dir $DATA_DIR --inp test_data.txt --type tail --num_frequent 5
python xt_input.py --data_dir $DATA_DIR --inp train_data_thorough.txt --type head --num_frequent 5
python xt_input.py --data_dir $DATA_DIR --inp train_data_thorough.txt --type tail --num_frequent 5
python xt_input.py --data_dir $DATA_DIR --inp validation_data_linked.txt --type head --num_frequent 5
python xt_input.py --data_dir $DATA_DIR --inp validation_data_linked.txt --type tail --num_frequent 5

mkdir data/$DATA_DIR/xt_models

# fasttext without pretraining
./extremetext supervised -input data/$DATA_DIR/train_data_thorough.txt.tail.xt -output data/$DATA_DIR/xt_models/tail_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -wordNgrams 2
./extremetext supervised -input data/$DATA_DIR/train_data_thorough.txt.head.xt -output data/$DATA_DIR/xt_models/head_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -wordNgrams 2

# fasttext with pre-trained vectors
./extremetext supervised -input data/$DATA_DIR/train_data_thorough.txt.tail.xt -output data/$DATA_DIR/xt_models/tail_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -wordNgrams 2 -pretrainedVectors data/olpbench/fastText_models/wiki-news-300d-1M-subword.vec
./extremetext supervised -input data/$DATA_DIR/train_data_thorough.txt.head.xt -output data/$DATA_DIR/xt_models/head_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -wordNgrams 2 -pretrainedVectors data/olpbench/fastText_models/wiki-news-300d-1M-subword.vec

# extreme text
./extremetext supervised -input data/$DATA_DIR/train_data_thorough.txt.tail.xt -output data/$DATA_DIR/xt_models/tail_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2
./extremetext supervised -input data/$DATA_DIR/train_data_thorough.txt.head.xt -output data/$DATA_DIR/xt_models/head_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2

MODEL=thorough_f5_d300
FILE=$DATA_DIR/test_data.txt
THREADS=2

cd data
../extremetext predict-prob $DATA_DIR/xt_models/head_$MODEL.bin $FILE.head.xtp 50 0 $FILE.head_$MODEL'.preds' $THREADS
cat $FILE.head_$MODEL'.preds'.0* > $FILE.head_$MODEL'.preds_parallel'
rm $FILE.head_$MODEL'.preds'.0*
mv $FILE.head_$MODEL'.preds_parallel' $FILE.head_$MODEL'.preds'

../extremetext predict-prob $DATA_DIR/xt_models/tail_$MODEL.bin $FILE.tail.xtp 50 0 $FILE.tail_$MODEL'.preds' $THREADS
cat $FILE.tail_$MODEL'.preds'.0* > $FILE.tail_$MODEL'.preds_parallel'
rm $FILE.tail_$MODEL'.preds'.0*
mv $FILE.tail_$MODEL'.preds_parallel' $FILE.tail_$MODEL'.preds'

python xt_output.py --inp $FILE --model $MODEL --type head --data $DATA_DIR
python xt_output.py --inp $FILE --model $MODEL --type tail --data $DATA_DIR