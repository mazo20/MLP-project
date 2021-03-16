import torch
from datasets import *

def normalisatonValues(dataset):
    loader = data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=8)
    
    d = next(iter(loader))[0]
    
    for i in range(3):
        print("Channel " + str(i))
        print("Mean: " + str(torch.mean(d[:, i, :, :])))
        print("Std: " + str(torch.std(d[:, i, :, :])))