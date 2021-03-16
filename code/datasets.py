import sys
import os
import images2chips
from PIL import Image
import ext_transforms as et
import numpy as np
from torch.utils import data
import torchvision
import torchvision.transforms as transforms


URLS = {
    'dataset-sample' : 'https://dl.dropboxusercontent.com/s/h8a8kev0rktf4kq/dataset-sample.tar.gz?dl=0',
    'dataset-medium' : 'https://dl.dropboxusercontent.com/s/r0dj9mhyv4bgbme/dataset-medium.tar.gz?dl=0',
}

def download_dataset(root, dataset):
    """ Download a dataset, extract it and create the tiles """
    
    if not os.path.exists(root):
        os.mkdir(root)
        
    dataset_dir = f'{root}/{dataset}'

    if dataset not in URLS:
        print(f"unknown dataset {dataset}")
        sys.exit(0)

    filename = f'{dataset}.tar.gz'
    url = URLS[dataset]

    if not os.path.exists(dataset_dir):
        
        print(f'downloading dataset "{dataset}"')
        os.system(f'curl "{url}" -o {filename}')
    else:
        print(f'zipfile "{filename}" already exists, remove it if you want to re-download.')

    if not os.path.exists(dataset_dir):
        print(f'extracting "{filename}"')
        os.system(f'tar -xvf {filename} -C {root}')
    else:
        print(f'folder "{dataset_dir}" already exists, remove it if you want to re-create.')
        
    image_chips = f'{dataset_dir}/image-chips'
    label_chips = f'{dataset_dir}/label-chips'
    elevation_chips = f'{dataset_dir}/elevation-chips'
    if not os.path.exists(image_chips) and not os.path.exists(label_chips) and not os.path.exists(elevation_chips):
        print("creating chips")
        images2chips.run(dataset_dir)
    else:
        print(f'chip folders "{image_chips}" and "{label_chips}" already exist, remove them to recreate chips.')
        
    if os.path.exists(filename):
        os.remove(filename)
        
def get_dataset(args):
    download_dataset(args.data_root, args.dataset)
    
    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(args.crop_size, args.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.5920, 0.5707, 0.5082],
                        std=[0.2663, 0.2328, 0.2338]),
    ])
    val_transform = et.ExtCompose([
        et.ExtResize(args.crop_size),
        et.ExtCenterCrop(args.crop_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.6045, 0.6181, 0.5966],
                        std=[0.1589, 0.1439, 0.1654]),
    ])
    
    train_dst = DroneDeploy(root=args.data_root, dataset=args.dataset, image_set='train', transform=train_transform)
    val_dst = DroneDeploy(root=args.data_root, dataset=args.dataset, image_set='val', transform=val_transform)
    
    return train_dst, val_dst

def dd_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap
    
class DroneDeploy(data.Dataset):
    cmap = dd_cmap()
    def __init__(self, root, dataset, image_set='train', transform=None):
        
        self.root = root
        self.dataset = dataset
        self.image_set = image_set
        self.transform = transform
        self.data_root = os.path.join(self.root, self.dataset)
        
        image_dir = os.path.join(self.data_root, 'image-chips')
        mask_dir = os.path.join(self.data_root, 'label-chips')

        if not os.path.isdir(image_dir):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if image_set == 'train':
            split_f = os.path.join(self.data_root, 'train.txt')
        elif image_set == 'val':
            split_f = os.path.join(self.data_root, 'valid.txt')
        else:
            split_f = os.path.join(self.data_root, 'test.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="val" or image_set="test"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x) for x in file_names]
        self.masks = [os.path.join(mask_dir, x) for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index])
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.images)
    
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]