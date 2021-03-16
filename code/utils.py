import torch
from datasets import *
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import torch.nn.functional as F

def save_images(loader, image, target, pred, denorm, img_id, root):
    root = root + '/images'
    if not os.path.exists(root):
        os.mkdir(root)
        
    image = image.detach().cpu().numpy()
    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
    target = loader.dataset.decode_target(target).astype(np.uint8)
    pred = loader.dataset.decode_target(pred).astype(np.uint8)

    Image.fromarray(image).save('%s/%d_image.png' % (root, img_id))
    Image.fromarray(target).save('%s/%d_target.png' % (root, img_id))
    Image.fromarray(pred).save('%s/%d_pred.png' % (root, img_id))

    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.imshow(pred, alpha=0.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.savefig('%s/%d_overlay.png' % (root, img_id), bbox_inches='tight', pad_inches=0)
    plt.close()
    
def create_result(opts):
    path = f'{opts.results_root}/{opts.mode}_{opts.model}_os_{opts.output_stride}_{opts.crop_size}_{opts.random_seed}.csv'
    
    if not os.path.exists(path):
        with open(path, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',)
            spamwriter.writerow(['model='+opts.model, 'os='+str(opts.output_stride), 
                                'crop='+str(opts.crop_size)])
            spamwriter.writerow(['Overall_Acc', 'Mean_Acc', 'FreqW_Acc', 'Mean_IoU'])
    
def save_result(score, opts):
    path = f'{opts.results_root}/{opts.mode}_{opts.model}_os_{opts.output_stride}_{opts.crop_size}_{opts.random_seed}.csv'
    with open(path, 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',)
        spamwriter.writerow([score['Overall Acc'], score['Mean Acc'], score['FreqW Acc'], score['Mean IoU']])

def normalisatonValues(dataset):
    loader = data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=8)
    
    d = next(iter(loader))[0]
    
    for i in range(3):
        print("Channel " + str(i))
        print("Mean: " + str(torch.mean(d[:, i, :, :])))
        print("Std: " + str(torch.std(d[:, i, :, :])))
        
class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)
    
def save_ckpt(path, opts, model, optimizer, scheduler, best_score, epoch):
        """ save current model
        """
        
        path = path + '/%s_%s_os%d_%d.pth' % (opts.model, opts.dataset, opts.output_stride, opts.random_seed)
        
        torch.save({
            "epoch": epoch,
            "best_score": best_score,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, path)
        print("Model saved as %s" % path)