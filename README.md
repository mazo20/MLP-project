# MLP-project

## How to run experiments on the cluster

1. SSH into a DICE machine
2. Run ```ssh mlp``` to connect to the cluster
3. Only the head node on the cluster has access to the internet
4. Install your conda environment with all of the required plugins and name it **mlp**
5. Clone the repo to /git directory. The *segmentation_arrayjob.sh* expects the project at git/MLP-project
6. Load the *cluster scripts*

```
echo 'export PATH=/home/$USER/git/cluster-scripts:$PATH' >> ~/.bashrc
echo 'export PATH=/home/$USER/git/cluster-scripts/experiments:$PATH' >> ~/.bashrc
source ~/.bashrc
```
7. On the head node run `python main.py --dataset dataset-sample` to download the dataset-sample
8. *segmentation_arrayjob.sh* will be run on each node and copy the data *if neccessary* using rsync to the scratch disk of each node
9. Use gen_exp.py to generate experiments.txt
10. Use run_exp.sh to run the experiments.txt on 2080Ti with 4 CPUs
11. `run_exp.sh` puts all experiments from the txt in the queue
