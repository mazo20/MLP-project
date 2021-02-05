import sys
import os
import images2chips
from PIL import Image
import ext_transforms as et

from torch.utils import data
import torchvision
import torchvision.transforms as transforms


URLS = {
    'dataset-sample' : 'https://dl.dropboxusercontent.com/s/h8a8kev0rktf4kq/dataset-sample.tar.gz?dl=0',
    'dataset-medium' : 'https://dl.dropboxusercontent.com/s/r0dj9mhyv4bgbme/dataset-medium.tar.gz?dl=0',
}

def download_dataset(dataset):
    """ Download a dataset, extract it and create the tiles """

    if dataset not in URLS:
        print(f"unknown dataset {dataset}")
        sys.exit(0)

    filename = f'{dataset}.tar.gz'
    url = URLS[dataset]

    if not os.path.exists(dataset):
        print(f'downloading dataset "{dataset}"')
        os.system(f'curl "{url}" -o {filename}')
    else:
        print(f'zipfile "{filename}" already exists, remove it if you want to re-download.')

    if not os.path.exists(dataset):
        print(f'extracting "{filename}"')
        os.system(f'tar -xvf {filename}')
    else:
        print(f'folder "{dataset}" already exists, remove it if you want to re-create.')
        
    image_chips = f'{dataset}/image-chips'
    label_chips = f'{dataset}/label-chips'
    if not os.path.exists(image_chips) and not os.path.exists(label_chips):
        print("creating chips")
        images2chips.run(dataset)
    else:
        print(f'chip folders "{image_chips}" and "{label_chips}" already exist, remove them to recreate chips.')
        
    if os.path.exists(filename):
        os.remove(filename)
        
def get_dataset(args):
    download_dataset(args.dataset)
    
    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(args.crop_size, args.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    val_transform = et.ExtCompose([
        et.ExtResize(args.crop_size),
        et.ExtCenterCrop(args.crop_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    
    train_dst = DroneDeploy(root=args.data_root, image_set='train', transform=train_transform)
    val_dst = DroneDeploy(root=args.data_root, image_set='val', transform=val_transform)
    
    return train_dst, val_dst
    
class DroneDeploy(data.Dataset):
    def __init__(self, root, image_set='train', transform=None):
        
        self.root = root
        self.image_set = image_set
        self.transform = transform
        
        image_dir = os.path.join(self.root, 'image-chips')
        mask_dir = os.path.join(self.root, 'label_chips')

        if not os.path.isdir(image_dir):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if image_set == 'train':
            split_f = os.path.join(self.root, 'train.txt')
        elif image_set == 'val':
            split_f = os.path.join(self.root, 'valid.txt')
        else:
            split_f = os.path.join(self.root, 'test.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="val" or image_set="test"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [image_dir for x in file_names]
        self.masks = [mask_dir for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.images)