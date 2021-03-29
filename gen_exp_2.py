"""Script for generating experiments.txt"""
import os
import sys
import itertools

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/dronedeploy/datasets/'
base_call = (f"python code/main.py  --data_root {DATA_HOME}")

config = {
    '--dataset':        ['dataset-medium'],
    '--pretrained':     ['true', 'false'],
    '--depth_mode':     ['dconv'],
    '--first_aware':    ['true', 'false'],
    '--model':          ['v3_resnet50'],
    '--lr':             ['0.01'],
    '--random_seed':    ['69'],
    '--results_root':   ['results/' + sys.argv[1]],

}

keys, values = zip(*config.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

nr_expts = len(permutations_dicts)
nr_servers = 10
avg_expt_time = 5  # hours
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {((round(nr_expts / nr_servers) + 1) * avg_expt_time)} hrs')

output_file = open("experiment.txt", "w")

for i, dictionary in enumerate(permutations_dicts):
    call = base_call
    for key, value in dictionary.items():
        call += " " + key + " " + value
    call += " " + "--index" + " " + str(i)
    print(call, file=output_file)
    
output_file.close()