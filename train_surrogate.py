import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from datasets import ImgDirDataset, LatentDataset
from torch.utils.data import DataLoader
from models.surrogate_models import ResNet_18

def setup_dataset(args):
    os.listdir(args.wm_img_folder) + os.listdir(args.unwm_img_folder)
    train_idxs, test_idxs = train_test_split()
    if args.mode == 'rawpix':
        train_split = ImgDirDataset([args.wm_img_folder,args.unwm_img_folder],[0,1], train_idxs)
        test_split = ImgDirDataset([args.wm_img_folder,args.unwm_img_folder],[0,1], test_idxs)
    elif args.mode == 'latent':
        train_split = LatentDataset([args.wm_img_folder,args.unwm_img_folder],[0,1], train_idxs)
        test_split = LatentDataset([args.wm_img_folder,args.unwm_img_folder],[0,1], test_idxs)
    elif args.mode == 't49latent':
        train_split = LatentDataset([args.wm_img_folder,args.unwm_img_folder],[0,1], train_idxs)
        test_split = LatentDataset([args.wm_img_folder,args.unwm_img_folder],[0,1], test_idxs)
    else:
        raise NotImplementedError(f"using unknwon data representation: {args.mode}")
    
    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_split, batch_size=args.batch_size, shuffle=True)

    return train_loader, test_loader

def setup_surrogate(args):
    ### Only two classes watermarked vs non-watermarked
    num_classes = 2

    if args.mode == 'rawpix':
        model_in_dim = 3
        data_type = None
    else:
        if args.model_id in ['stabilityai/stable-diffusion-2-1-base','stabilityai/sdxl-vae']:
            model_in_dim = 4
            data_type = torch.cfloat
        elif args.model_id in ['ostris/vae-kl-f8-d16']:
            model_in_dim = 16
            data_type = torch.cfloat
        else:
            raise f"Model type for {args.model_id} not implemented."
        
    model = ResNet_18(model_in_dim, num_classes, dtype=data_type).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    return model, criterion, optimizer

def main(args):
    train_data, test_data = setup_dataset(args)
    model, criterion, optimizer = setup_surrogate(args)

    best_val_acc = 0.0
    best_model_state = None
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Adversairal surrogate")
    parser.add_argument('wm_img_folder', type=str)
    parser.add_argument('unwm_img_folder', type=str)
    parser.add_argument('model_save_path', type=str)
    parser.add_argument('model_save_name', type=str)

    parser.add_argument('--mode', default = 'rawpix', choices = ['rawpix','latent','t49latent'], type=str)
    parser.add_argument('--vae', default = 'stabilityai/stable-diffusion-2-1-base',
                        choices = ['stabilityai/stable-diffusion-2-1-base','stabilityai/sdxl-vae','ostris/vae-kl-f8-d16'],
                        type = str
    )
    parser.add_argument('--use_vae', default=False, type=bool)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--seed', default=999999, type=int)

    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()
    main(args)
