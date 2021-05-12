import sys
import os
# sys.path.append(sys.path[0]+"/../")
import argparse
import logging
import os
import pickle
import pprint
import sys
import time
import numpy as np
import random
import torch
from tqdm import tqdm
import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import ast

def read_mentions(path):
    mapp = {}
    mentions = []
    lines = open(path,'r').readlines()
    for line in tqdm(lines[1:]): 
        line = line.strip().split("\t")
        mentions.append(line[0])
        mapp[line[0]] = len(mapp)
    return mentions,mapp

class kb(object):
    """
        stores data about knowledge base
        if split_as_regular_sentences is True:
            triples stores as [["this","is","a","ball"],...] (i.e. not as e1,r,e2)
    """
    def __init__(self,filename,em_map=None,rm_map=None,split_as_regular_sentences = False):
        print("Reading file {}".format(filename))
        lines = open(filename,'r').readlines()
        self.triples = []
        self.e1_all_answers = [] # saves all answers for ith triple using their 'int' value from map 
        self.e2_all_answers = []
        self.em_map = em_map
        self.rm_map = rm_map

        #save alternate mentions, triple wise
        for line in tqdm(lines):
            line = line.strip().split("\t")
            if (not split_as_regular_sentences):
                self.triples.append([line[0],line[1],line[2]])
                e1_answers = line[3].split("|||")
                e2_answers = line[4].split("|||")
                if(em_map!=None):
                    mapped_e1_answers = []
                    mapped_e2_answers = []

                    for i in range(len(e1_answers)):
                        if(e1_answers[i] not in em_map): # sometimes the answer isn't present in em_map. ignoring that answer
                            continue
                        mapped_e1_answers.append(em_map[e1_answers[i]])
                    for i in range(len(e2_answers)):
                        if(e2_answers[i] not in em_map): # sometimes the answer isn't present in em_map. ignoring that answer
                            continue
                        mapped_e2_answers.append(em_map[e2_answers[i]])
                    self.e1_all_answers.append(mapped_e1_answers)
                    self.e2_all_answers.append(mapped_e2_answers)
                else:
                    self.e1_all_answers.append(e1_answers)
                    self.e2_all_answers.append(e2_answers)
            else:
                self.triples.append(line[0].split()+line[1].split()+line[2].split())
        if(not split_as_regular_sentences):
            self.triples = np.array(self.triples)



has_cuda = torch.cuda.is_available()

def main():
    data_dir = "olpbench"
    freq_r_tail = {}
    freq_r_head = {}
    entity_mentions,em_map = read_mentions(os.path.join(data_dir,"mapped_to_ids","entity_id_map.txt"))
    _,rm_map = read_mentions(os.path.join(data_dir,"mapped_to_ids","relation_id_map.txt"))

    # train_kb = kb(os.path.join(data_dir,"test_data.txt"), em_map = None, rm_map = None)
    train_kb = kb(os.path.join(data_dir,"train_data_thorough.txt"), em_map = None, rm_map = None)
    for triple in tqdm(train_kb.triples, desc="getting r freq"):
        e1 = triple[0].item()
        r  = triple[1].item()
        e2 = triple[2].item()
        if r not in freq_r_tail:
            freq_r_tail[r] = {}
        if em_map[e2] not in freq_r_tail[r]:
            freq_r_tail[r][em_map[e2]] = 0
        freq_r_tail[r][em_map[e2]] += 1

        if r not in freq_r_head:
            freq_r_head[r] = {}
        if em_map[e1] not in freq_r_head[r]:
            freq_r_head[r][em_map[e1]] = 0
        freq_r_head[r][em_map[e1]] += 1

    f = open("olpbench/r-freq_top100_thorough_head.pkl","wb")
    final_data = {}
    for r in freq_r_head:
        final_list = list(zip(list(freq_r_head[r].values()),list(freq_r_head[r].keys())))
        final_list.sort(reverse=True)
        final_list = final_list[:100]
        final_data[r] = final_list
    pickle.dump(final_data,f)
    f.close()

    f = open("olpbench/r-freq_top100_thorough_tail.pkl","wb")
    final_data = {}
    for r in freq_r_tail:
        final_list = list(zip(list(freq_r_tail[r].values()),list(freq_r_tail[r].keys())))
        final_list.sort(reverse=True)
        final_list = final_list[:100]
        final_data[r] = final_list
    pickle.dump(final_data,f)
    f.close()

if __name__=="__main__":
    main()


