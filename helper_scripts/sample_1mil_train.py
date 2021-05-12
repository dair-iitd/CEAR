import os
from tqdm import tqdm
import random

data_dir = "olpbench"
output_path = "olpbench/train_data_thorough_1mil.txt"
f = open(output_path,'w')
lines = open(os.path.join(data_dir,"train_data_thorough.txt"),'r').readlines()
for line in tqdm(lines):
	if random.random()<1/28:
		f.write(line)
