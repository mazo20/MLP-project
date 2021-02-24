import argparse
import torch
import cv2
from tqdm import tqdm
import numpy as np
from datasets import *
from utils import *
from scheduler import *
from stream_metrics import *
import network

model_map = {
    'v3plus_resnet50': network.deeplabv3plus_resnet50,
    'v3plus_resnet101': network.deeplabv3plus_resnet101,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default='datasets', help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='dataset-sample', choices=['dataset-sample', 'dataset-medium'])
    parser.add_argument('--model', type=str, default='v3plus_resnet50', choices=['v3plus_resnet50', 'v3plus_resnet101'], help="model name")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    
    parser.add_argument("--batch_size", type=int, default=2, help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=8, help='batch size for validation (default: 4)')
    parser.add_argument('--crop_size', type=int, default=100, help="size of the crop size  during transform")
    parser.add_argument('--num_classes', type=int, default=7, help="number of the classes")
    parser.add_argument('--output_stride', type=int, default=16, help="output stride of the image, default: 16")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--total_epochs", type=int, default=30, help="Number of epochs per training")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    
    
    args = parser.parse_args()
    
    train_dst, val_dst = get_dataset(args)
    train_loader = data.DataLoader(train_dst, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = data.DataLoader(val_dst, batch_size=args.val_batch_size, shuffle=True, num_workers=8)
    
    '''
    Use to print normalisation values (mean, std) for the given dataset
    '''
    #normalisatonValues(train_dst)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    
    
    for image, target in train_loader:
        img = (image[0].numpy() * 255).transpose(1, 2, 0).astype(np.uint8)
        Image.fromarray(img).save('img.png')
        break
    
    print("Train set: %d, Val set: %d" % (len(train_dst), len(val_dst)))
    
    metrics = StreamSegMetrics(args.num_classes)
    
    model = model_map[args.model](num_classes=args.num_classes, output_stride=args.output_stride)
    
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = PolyLR(optimizer, args.total_epochs * len(val_loader), power=0.9)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    for epoch in tqdm(range(args.total_epochs)):
        
        model.train()
        metrics.reset()
        pbar = tqdm(train_loader)
        for images, labels in pbar:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)[:,:,:]
            
            
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            # print(preds[0,:20,:20])
            # print(targets[0,:20,:20])
            
            metrics.update(targets, preds)
            score = metrics.get_results()
            pbar.set_postfix({"IoU": score["Mean IoU"]})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # score = validate(model, optimizer, scheduler, best_score, cur_epochs)
    
    