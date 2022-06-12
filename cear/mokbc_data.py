import ipdb
import pickle
import os
import json
import ast
import sys

dir_ = sys.argv[1]
# 'open_kbc_data/simkgc_en/'

files = {'test': 'test_{}.json', 'train': 'train_{}.json', 'valid': 'val_{}.json'}

entities = set()
relations = set()
for mode in ['train', 'valid', 'test']:
    jfil = json.load(open(dir_+files[mode].format('forward'), encoding='utf-8'))
    for triple in jfil:
        entities.add(triple['head'])
        entities.add(triple['tail'])
        relations.add(triple['relation'])
    jfil = json.load(open(dir_+files[mode].format('backward'), encoding='utf-8'))
    for triple in jfil:
        entities.add(triple['head'])
        entities.add(triple['tail'])
        relations.add(triple['relation'].lstrip('inverse').strip())

print('Total number of entities, relations = ', len(entities), len(relations))
entities = list(entities)
relations = list(relations)
entityD = {}
relationD = {}

# ipdb.set_trace()

with open(dir_+'/entities.dict','w', encoding='utf-8') as ent_f, open(dir_+'/entity2text.txt','w', encoding='utf-8') as ent2txt_f,\
    open(dir_+'/relations.dict','w', encoding='utf-8') as rel_f, open(dir_+'/relation2text.txt','w', encoding='utf-8') as rel2txt_f:
    for eid, entity in enumerate(entities):
        entityD[entity] = eid
        ent_f.write(str(eid)+'\t'+entity+'\n')
        ent2txt_f.write(entity+'\t'+entity+'\n')
    for rid, relation in enumerate(relations):
        relationD[relation] = rid
        rel_f.write(str(rid)+'\t'+relation+'\n')
        rel2txt_f.write(relation+'\t'+relation+'\n')

for mode in ['train', 'valid', 'test']:
    jf = json.load(open(dir_+files[mode].format('forward'), encoding='utf-8'))
    jb = json.load(open(dir_+files[mode].format('backward'), encoding='utf-8'))

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
        finalD[(head_id, rel_id, tail_id)]['tail-batch'] = {'index': eids, 'confidence': scores, 'bias': [tail_id], 
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
        finalD[(head_id, rel_id, tail_id)]['head-batch'] = {'index': eids, 'confidence': scores, 'bias': [head_id], 
        'score': {'MR': example['rank'], 'HITS1': float(example['rank']==1), 
        'HITS3': float(example['rank']<=3), 'HITS10': float(example['rank']<=10)}}

    if mode == 'train':
        fp = 'train_data.pkl'
    if mode == 'valid':
        fp = 'validation_data.pkl'
    if mode == 'test':
        fp = 'test_data.pkl'
    pickle.dump(finalD, open(dir_+fp, 'wb'))
            
