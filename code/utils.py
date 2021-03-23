import os
import csv
import torch
import matplotlib
import numpy               as np
import matplotlib.pyplot   as plt
import torch.nn.functional as F

from PIL               import Image
from config            import LABELMAP, ELEVATION_IGNORE
from datasets          import *
from matplotlib        import cm
from matplotlib.colors import ListedColormap

def save_images(loader, image, target, pred, denorm, img_id, root):
    root = root + '/images'
    if not os.path.exists(root):
        os.mkdir(root)
        
    image  = image.detach().cpu().numpy()
    image  = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
    target = label_to_cmap(target)
    pred   = label_to_cmap(pred)

    Image.fromarray(image).save( '%s/%d_image.png'  % (root, img_id))
    Image.fromarray(target).save('%s/%d_target.png' % (root, img_id))
    Image.fromarray(pred).save(  '%s/%d_pred.png'   % (root, img_id))

    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.imshow(pred, alpha=0.5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.savefig('%s/%d_overlay.png' % (root, img_id), bbox_inches='tight', pad_inches=0)
    plt.close()

# cmap_name is a String name of color map from https://matplotlib.org/stable/tutorials/colors/colormaps.html
# Converts raw elevation encoding into pretty color map. Saves it in out_file.
def elevation_to_cmap(in_file, out_file, cmap_name='magma'):
    color_map = cm.get_cmap(cmap_name)
    image     = cv2.imdecode(np.fromfile(in_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    real_min  = np.min(image[image != ELEVATION_IGNORE])
    real_max  = np.max(image)
    
    # Encodes the IGNORE class with same elevation as minium non-IGNORE pixel.
    image[image == ELEVATION_IGNORE] = real_min
    
    # Normalize and convert to color map.
    image = (image - real_min) / (real_max - real_min)
    image = color_map(image)[:,:,:3]
    image = image * 255
    image = np.uint8(image)
    image = Image.fromarray(image)
    
    image.save(out_file)

# Converts raw label map into pretty color map. Returns 3D tensor of RGB values.
def label_to_cmap(label_map):
    cmap      = get_custom_cmap()
    label_map = label_map / 6
    label_map = cmap(label_map)[:,:,:3]
    label_map = label_map * 255
    label_map = np.uint8(label_map)
    
    return label_map

# Creates instance of matplotlib color map with label colors from our dataset.
def get_custom_cmap():
    cmap        = cm.get_cmap('viridis', 7)
    custom_cmap = cmap(np.linspace(0, 1, 7))

    for i in range(6):
        custom_cmap[i] = np.array(LABELMAP[i+1] + (255,)) / 255

    custom_cmap[6] = np.array(LABELMAP[0] + (255,)) / 255
    custom_cmap    = ListedColormap(custom_cmap)

    return custom_cmap
    
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
        print("Mean:   " + str(torch.mean(d[:, i, :, :])))
        print("Std:    " + str(torch.std(d[:, i, :, :])))
        
class Denormalize(object):
    def __init__(self, mean, std):
        mean       = np.array(mean)
        std        = np.array(std)
        self._mean = -mean/std
        self._std  = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)
    
def save_ckpt(path, opts, model, optimizer, scheduler, best_score, epoch):
        """ save current model
        """
        
        root = os.path.join(path, 'output', opts.results_root)
        if not os.path.exists(root):
            os.mkdir(root)
        
        path = root + '/%s_%s_os%d_%d.pth' % (opts.model, opts.dataset, opts.output_stride, opts.random_seed)
        
        torch.save({
            "epoch":           epoch,
            "best_score":      best_score,
            "model_state":     model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, path)
        print("Model saved as %s" % path)