# MLP-project

## How to run experiments on the cluster

1. SSH into a DICE machine
2. Run ```ssh mlp``` to connect to the cluster
3. Only the head node on the cluster has access to the internet
4. Install your conda environment with all of the required plugins and name it **mlp**
5. Clone the repo to /git directory. The *segmentation_arrayjob.sh* expects the project at git/MLP-project
6. ```echo 'export PATH=/home/$USER/git/cluster-scripts:$PATH' >> ~/.bashrc
echo 'export PATH=/home/$USER/git/cluster-scripts/experiments:$PATH' >> ~/.bashrc
source ~/.bashrc
```s

