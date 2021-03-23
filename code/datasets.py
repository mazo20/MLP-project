import os
import sys
import torchvision
import images2chips
import numpy                  as np
import ext_transforms         as et
import torchvision.transforms as transforms

from PIL         import Image
from config      import ELEVATION_IGNORE
from torch.utils import data


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
        et.ExtRandomScale((0.8, 1.2)),
        et.ExtRandomCrop(size=(args.crop_size, args.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.5220, 0.5120, 0.4516, 0.09183968857583202],
                        std =[0.1983, 0.1882, 0.1934, 0.05884465529577935]),
    ])
    val_transform = et.ExtCompose([
        et.ExtResize(args.crop_size),
        et.ExtCenterCrop(args.crop_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.5220, 0.5120, 0.4516, 0.09183968857583202],
                        std =[0.1983, 0.1882, 0.1934, 0.05884465529577935]),
    ])
    
    train_dst = DroneDeploy(root=args.data_root, dataset=args.dataset, image_set='train', transform=train_transform)
    val_dst   = DroneDeploy(root=args.data_root, dataset=args.dataset, image_set='val',   transform=val_transform)
    
    return train_dst, val_dst
    
class DroneDeploy(data.Dataset):
    def __init__(self, root, dataset, image_set='train', transform=None):
        
        self.root      = root
        self.dataset   = dataset
        self.image_set = image_set
        self.transform = transform
        self.data_root = os.path.join(self.root, self.dataset)
        
        image_dir = os.path.join(self.data_root, 'image-chips')
        mask_dir  = os.path.join(self.data_root, 'label-chips')
        eleva_dir = os.path.join(self.data_root, 'elevation-chips')

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
        
        self.images = [os.path.join(image_dir, x)              for x in file_names]
        self.masks  = [os.path.join(mask_dir, x)               for x in file_names]
        self.eleva  = [os.path.join(eleva_dir, x[:-3] + 'tif') for x in file_names]

        assert (len(self.images) == len(self.masks))
        assert (len(self.images) == len(self.eleva))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img    = Image.open(self.images[index])
        target = Image.open(self.masks[index])
        eleva  = Image.open(self.eleva[index])
        eleva  = np.array(eleva)

        # Impute IGNORE elevation values with mean.
        eleva[eleva == ELEVATION_IGNORE] = 11.553864551722832

        four_chan         = np.zeros(eleva.shape + (4,))
        four_chan[:,:,:3] = np.array(image)
        four_chan[:,:,3]  = ((eleva + 39.504932) / (39.504932 + 504.8893)) * 255
        img               = Image.fromarray(np.uint8(four_chan))

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.images)