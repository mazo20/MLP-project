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
    '--output_stride': ['16'],
    '--dataset': ['dataset-medium'],
    #'--batch_size': ['8', '16'],
    '--crop_size': ['300'],
    #'--vertical_flip': ['true', 'false'],
    #'--horizontal_flip': ['true', 'false'],
    '--min_scaling': ['0.5', '0.8', '1'],
    '--max_scaling': ['1', '1.25', '2'],
    '--results_root': ['results/' + sys.argv[1]],
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
    call += " " + "--random_seed" + " " + str(i)
    print(call, file=output_file)
    
output_file.close()
