import argparse
import torch
import cv2
from tqdm import tqdm
import numpy as np
from datasets import *
import utils as utils
from scheduler import *
from stream_metrics import *
import network

model_map = {
    'v3plus_resnet50': network.deeplabv3plus_resnet50,
    'v3plus_resnet101': network.deeplabv3plus_resnet101,
}

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./datasets', help="path to Dataset")
    parser.add_argument("--results_root", type=str, default='./results')
    parser.add_argument("--dataset", type=str, default='dataset-sample', choices=['dataset-sample', 'dataset-medium'])
    parser.add_argument('--model', type=str, default='v3plus_resnet50', choices=['v3plus_resnet50', 'v3plus_resnet101'], help="model name")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--save_val_results", action='store_true', default=False, help="save segmentation results to \"./results\"")
    parser.add_argument("--mode", default='train', choices=['train', 'validate'])
    parser.add_argument("--random_seed", type=int, default=0)
    
    parser.add_argument("--ckpt", default=None, type=str,help="restore from checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=8, help='batch size for validation (default: 4)')
    parser.add_argument('--crop_size', type=int, default=100, help="size of the crop size  during transform")
    parser.add_argument('--num_classes', type=int, default=7, help="number of the classes")
    parser.add_argument('--output_stride', type=int, default=16, help="output stride of the image, default: 16")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--total_epochs", type=int, default=30, help="Number of epochs per training")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    
    args = parser.parse_args()
    return args

def validate(model):
    model.eval()
    metrics.reset()
    
    if args.save_val_results:
        denorm = utils.Denormalize(mean=[0.6045, 0.6181, 0.5966],
                                    std=[0.1589, 0.1439, 0.1654])
        img_id = 0  

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs, ints = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            
            metrics.update(targets, preds)
            
            if args.save_val_results:
                for i in range(len(images)):
                    at_maps = [ints[j][i] for j in range(len(ints))]
                    utils.save_images(val_loader, images[i], targets[i], preds[i], denorm, img_id, args.results_root)
                    img_id += 1 
                        
        score = metrics.get_results()
        
    model.train()
    return score

def main():
    best_score = 0.0
    epoch = 0
    
    model = model_map[args.model](num_classes=args.num_classes, output_stride=args.output_stride)
    
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*args.lr},
        {'params': model.classifier.parameters(), 'lr': args.lr},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = PolyLR(optimizer, args.total_epochs * len(val_loader), power=0.9)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
    print(args.ckpt)
    
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = torch.nn.DataParallel(model)
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        epoch = checkpoint.get("epoch", 0)
        best_score = checkpoint.get('best_score', 0.0)
        print("Model restored from %s" % args.ckpt)
        del checkpoint  # free memory 
    else:
        model = torch.nn.DataParallel(model)
        model.to(device)
    
    #Create results csv
    utils.create_result(args)
    
    if args.mode == 'validate':
        
        score = validate(model)
        print(metrics.to_str(score))
        return
    
    
    print(epoch)
    
    for epoch in tqdm(range(epoch, args.total_epochs)):
        
        model.train()
        metrics.reset()
        pbar = tqdm(train_loader)
        for images, labels in pbar:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            
            metrics.update(targets, preds)
            score = metrics.get_results()
            pbar.set_postfix({"IoU": score["Mean IoU"]})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        score = validate(model)
        print(metrics.to_str(score))
        utils.save_result(score, args)
        
        if score['Mean IoU'] > best_score:  # save best model
            best_score = score['Mean IoU']
            utils.save_ckpt(args.data_root, args, model, optimizer, scheduler, best_score, epoch+1)

if __name__ == '__main__':
    args = get_argparser()
    
    train_dst, val_dst = get_dataset(args)
    train_loader = data.DataLoader(train_dst, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = data.DataLoader(val_dst, batch_size=args.val_batch_size, shuffle=True, num_workers=8)
    
    '''
    Use to print normalisation values (mean, std) for the given dataset
    '''
    #utils.normalisatonValues(train_dst)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    
    print("Train set: %d, Val set: %d" % (len(train_dst), len(val_dst)))
    
    metrics = StreamSegMetrics(args.num_classes)
    
    main()
