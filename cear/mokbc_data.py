import ipdb
import pickle
import os
import json
import ast

dir_ = 'open_kbc_data/simkgc_en/'

files = {'test': 'test_{}.json', 'train': 'train_{}.json', 'valid': 'val_{}.json'}

entities = set()
relations = set()
for mode in ['train', 'valid', 'test']:
    jfil = json.load(open(dir_+files[mode].format('forward')))
    for triple in jfil:
        entities.add(triple['head'])
        entities.add(triple['tail'])
        relations.add(triple['relation'])
    jfil = json.load(open(dir_+files[mode].format('backward')))
    for triple in jfil:
        entities.add(triple['head'])
        entities.add(triple['tail'])
        relations.add(triple['relation'].lstrip('inverse').strip())

print('Total number of entities, relations = ', len(entities), len(relations))
entities = list(entities)
relations = list(relations)
entityD = {}
relationD = {}

with open(dir_+'/entities.dict','w') as ent_f, open(dir_+'/entity2text.txt','w') as ent2txt_f,\
    open(dir_+'/relations.dict','w') as rel_f, open(dir_+'/relation2text.txt','w') as rel2txt_f:
    for eid, entity in enumerate(entities):
        entityD[entity] = eid
        ent_f.write(str(eid)+'\t'+entity+'\n')
        ent2txt_f.write(entity+'\t'+entity+'\n')
    for rid, relation in enumerate(relations):
        relationD[relation] = rid
        rel_f.write(str(rid)+'\t'+relation+'\n')
        rel2txt_f.write(relation+'\t'+relation+'\n')

for mode in ['train', 'valid', 'test']:
    jf = json.load(open(dir_+files[mode].format('forward')))
    jb = json.load(open(dir_+files[mode].format('backward')))

    finalD = {}
    for example in jf:
        head_id = entityD[example['head']]
        rel_id = relationD[example['relation']]
        tail_id = entityD[example['tail']]
        eids, scores = [], []
        for entity, score in ast.literal_eval(example['topk_score_info']).items():
            assert entity in entityD, ipdb.set_trace()
            eids.append(entityD[entity])
            # else:
            #     eids.append(0)
            scores.append(score)
        finalD[(head_id, rel_id, tail_id)] = {}
        finalD[(head_id, rel_id, tail_id)]['head-batch'] = {'index': eids, 'confidence': scores, 'bias': [], 
        'score': {'MR': example['rank'], 'HITS1': float(example['rank']==1), 
        'HITS3': float(example['rank']<=3), 'HITS10': float(example['rank']<=10)}}

    for example in jb:
        head_id = entityD[example['tail']]
        rel_id = relationD[example['relation'].lstrip('inverse').strip()]
        tail_id = entityD[example['head']]
        assert (head_id, rel_id, tail_id) in finalD, ipdb.set_trace()
        eids, scores = [], []
        for entity, score in ast.literal_eval(example['topk_score_info']).items():
            assert entity in entityD, ipdb.set_trace()
            eids.append(entityD[entity])
            # else:
            #     eids.append(0)
            scores.append(score)
        finalD[(head_id, rel_id, tail_id)]['tail-batch'] = {'index': eids, 'confidence': scores, 'bias': [], 
        'score': {'MR': example['rank'], 'HITS1': float(example['rank']==1), 
        'HITS3': float(example['rank']<=3), 'HITS10': float(example['rank']<=10)}}

    if mode == 'train':
        fp = 'train_data.pkl'
    if mode == 'valid':
        fp = 'validation_data.pkl'
    if mode == 'test':
        fp = 'test_data.pkl'
    pickle.dump(finalD, open(dir_+fp, 'wb'))
            

# python convert_kbc.py --kge_output open_kbc_data/simkgc_en/test_data.pkl --kge_data_dir open_kbc_data/simkgc_en/ --output_dir open_kbc_data/simkgc_en/ --kbe_data_dir open_kbc_data/simkgc_en/ --output_file test_data.txt --model simkgc --entity_map --relation_map --filter --predictions --scores
# python convert_kbc.py --kge_output open_kbc_data/simkgc_en/validation_data.pkl --kge_data_dir open_kbc_data/simkgc_en/ --output_dir open_kbc_data/simkgc_en/ --kbe_data_dir open_kbc_data/simkgc_en/ --output_file validation_data.txt --filter_val --predictions --scores_val --model simkgc_en
# python convert_kbc.py --kge_output open_kbc_data/simkgc_en/train_data.pkl --kge_data_dir open_kbc_data/simkgc_en/ --output_dir open_kbc_data/simkgc_en/ --kbe_data_dir open_kbc_data/simkgc_en/ --output_file train_data.txt --predictions --model simkgc_en

# python tokenize_pkl.py --inp open_kbc_data/simkgc_en/mapped_to_ids/entity_id_map.txt --model_str bert-base-cased
# python tokenize_pkl.py --inp open_kbc_data/simkgc_en/mapped_to_ids/relation_id_map.txt --model_str bert-base-cased