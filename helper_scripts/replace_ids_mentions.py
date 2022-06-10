## For ReVerb20K and ReVerb45K datasets
import sys
import ipdb

data_dir = sys.argv[1]

def create_map(file_name):
    id_to_string_map = {}
    for i, line in enumerate(open(file_name,'r').readlines()):
        if i==0: # skip header
            continue
        line = line.strip().split('\t')
        id_to_string_map[line[1]] = line[0]
    return id_to_string_map

id_to_em_map = create_map(data_dir+'/mapped_to_ids/entity_id_map.txt')
id_to_rel_map = create_map(data_dir+'/mapped_to_ids/relation_id_map.txt')

def write_data(inp, out):
    outf = open(out,'w')
    for i, line in enumerate(open(inp).readlines()):
        line = line.strip().split('\t')
        e1, rel, e2 = line[0], line[1], line[2]
        if i<5:
            print(id_to_em_map[e1], id_to_rel_map[rel], id_to_em_map[e2])
        outf.write(f"{id_to_em_map[e1]}\t{id_to_rel_map[rel]}\t{id_to_em_map[e2]}\t{id_to_em_map[e1]}\t{id_to_em_map[e2]}\n")
    outf.close()

write_data(data_dir+'/train_trip.txt', data_dir+'/train_data_thorough.txt')
write_data(data_dir+'/valid_trip.txt', data_dir+'/validation_data_linked.txt')
write_data(data_dir+'/test_trip.txt', data_dir+'/test_data.txt')
