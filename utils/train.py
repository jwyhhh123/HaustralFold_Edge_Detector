import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import JaccardIndex
from torchvision import transforms
from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
#from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import datasets
from third_party.Autoencoder.models import SegNet
from third_party.FlowNet2.models import FlowNet2
from third_party.DexiNed.models import DexiNed
from third_party.DexiNed.losses import * #
from third_party.DexiNed.utils.image import save_image_batch_to_disk
from utils import warp
from utils.fuse import fuse
from utils.metric import QuadrupletLoss

transform = transforms.Compose([
    transforms.GaussianBlur(3)
])


''' Train DexiNed '''
def trainDexiNed(device, args):
    if args.closs == 'triplet':
        # train DexiNed with Triplet loss, using SegNet encoder
        trainTriplet(device, args)
    elif args.closs == 'quadruplet':
        # train DexiNed with Quadruplet loss, using SegNet encoder
        trainQuadruplet(device, args)


''' private functions only called inside this script'''

# Train DexiNed by Triplet loss + MSE/BDCN
def trainTriplet(device, args):
    # Parameters
    SAVE_PATH = os.path.join('checkpoints',args.checkpoint)
    IMG_TRAIN_PATH = os.path.join('dataset_origin','train','TC')
    IMG_VAL_PATH = os.path.join('dataset_origin','val','TC')

    LOAD_PATH_DEXI = os.path.join('checkpoints','triplet_mse_w10_best.pth')
    LOAD_PATH_AUTO = os.path.join('checkpoints','BEST_checkpoint.pth')

    # Hyperparameters
    batch_size = 3
    epsilon    = args.epsilon
    epochs     = args.epochs

    # Accelarate training by using cudnn benchmark
    if torch.cuda.is_available():
       torch.backends.cudnn.benchmark=True

    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"

    if not os.path.isfile(LOAD_PATH_DEXI):
       raise FileNotFoundError(
            f"Checkpoint file not found: {LOAD_PATH_DEXI}")
    elif not os.path.isfile(LOAD_PATH_AUTO):
       raise FileNotFoundError(
            f"Checkpoint file not found: {LOAD_PATH_AUTO}")
    print(f"Restoring weights from: {LOAD_PATH_DEXI}")
    print(f"Restoring weights from: {LOAD_PATH_AUTO}")

    dexi = DexiNed().to(device)
    auto = SegNet().to(device)
    dexi.load_state_dict(torch.load(LOAD_PATH_DEXI, map_location=device))
    auto.load_state_dict(torch.load(LOAD_PATH_AUTO, map_location=device))

    auto.eval()

    train_set = datasets.Dataset(IMG_TRAIN_PATH,
                          img_width=args.img_width,
                          img_height=args.img_height,
                          mean_bgr=args.mean_pixel_values[0:3] if len(
                              args.mean_pixel_values) == 4 else args.mean_pixel_values,
                          model=args.model,
                          mode = args.mode,
                          transformer=transform
                          )

    val_set = datasets.Dataset(IMG_VAL_PATH,
                          img_width=args.img_width,
                          img_height=args.img_height,
                          mean_bgr=args.mean_pixel_values[0:3] if len(
                              args.mean_pixel_values) == 4 else args.mean_pixel_values,
                          model=args.model,
                          mode = args.mode,
                          transformer=transform
                          )

    train_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

    optimizer = optim.RMSprop(dexi.parameters(), lr=epsilon, alpha=0.99, eps=1e-08)
    #scheduler = CosineAnnealingWarmRestarts(optimizer,T_0 = 1,T_mult=2,eta_min=0.000000001)

    tc_loss = nn.TripletMarginLoss(margin=args.margin1)

    if args.ploss == 'bdcn':
        pixel_wise_loss = bdcn_loss2
    elif args.ploss == 'mse':
        pixel_wise_loss = nn.MSELoss()

    # Define a scale tensor replicates mean_pixel_vaule with same dim of data
    norm_tensor = torch.zeros(3, args.img_width, args.img_height)
    norm_tensor[0] += args.mean_pixel_values[0]
    norm_tensor[1] += args.mean_pixel_values[1]
    norm_tensor[2] += args.mean_pixel_values[2]
    norm_tensor = norm_tensor.repeat(batch_size,1,1,1).to(device)

    trainlossTracker = []
    validlossTracker = []
    loss_min   = 999999

    print('Started training DexiNed with Triplet metric...')
    for epoch in range(epochs):
        trainLoss = 0
        valLoss   = 0
        successor = iter(train_set.video_frames)
        load = next(successor)
        num_frames = load['frames']
        startVideo = True
        last_two_frames = None
        last_two_labels = None
        for i, data in enumerate(train_loader):
            
            if (i+1)*batch_size <= num_frames:
                if startVideo:
                    startVideo = False
                    # Feed data into DexiNed
                    images = data['images'].to(device)
                    labels = data['labels'].to(device)
                    last_two_frames = images[-2:,:,:,:]
                    last_two_labels = labels[-2:,:,:,:]

                    trainLoss += _propagateTriplet(images, dexi, auto, tc_loss, args, device, labels=labels, pw_loss=pixel_wise_loss, opt=optimizer)
                else:
                    images = data['images'].to(device)
                    labels = data['labels'].to(device)

                    # anchor is 2nd of last batch
                    images_ = torch.cat((last_two_frames,torch.unsqueeze(images[0,:,:,:],0)),0)
                    labels_ = torch.cat((last_two_labels,torch.unsqueeze(labels[0,:,:,:],0)),0)
                    trainLoss += _propagateTriplet(images_, dexi, auto, tc_loss, args, device, labels=labels_, pw_loss=pixel_wise_loss, opt=optimizer)

                    # anchor is 3rd of last batch
                    images_ = torch.cat((torch.unsqueeze(last_two_frames[-1,:,:,:],0),images[0:2,:,:,:]),0)
                    labels_ = torch.cat((torch.unsqueeze(last_two_labels[-1,:,:,:],0),labels[0:2,:,:,:]),0)
                    trainLoss += _propagateTriplet(images_, dexi, auto, tc_loss, args, device, labels=labels_, pw_loss=pixel_wise_loss, opt=optimizer)

                    # anchor is 1st of this batch
                    trainLoss += _propagateTriplet(images, dexi, auto, tc_loss, args, device, labels=labels, pw_loss=pixel_wise_loss, opt=optimizer)

                    last_two_frames = images[-2:,:,:,:]
                    last_two_labels = labels[-2:,:,:,:]

            elif (i+1)*batch_size < len(train_set):
                load = next(successor)
                num_frames += load['frames']
                startVideo = True

        
        # Validation
        dexi.eval()
        with torch.no_grad():
            successor_val = iter(val_set.video_frames)
            load_val = next(successor_val)
            num_frames_val = load_val['frames']
            startVideo = True
            last_two_frames = None
            last_two_labels = None
            for j, data in enumerate(val_loader):

                if (j+1)*batch_size <= num_frames_val:
                    if startVideo:
                        startVideo = False
                        # Feed data into DexiNed
                        images = data['images'].to(device)
                        labels = data['labels'].to(device)
                        last_two_frames = images[-2:,:,:,:]
                        last_two_labels = labels[-2:,:,:,:]

                        valLoss += _propagateTriplet(images, dexi, auto, tc_loss, args, device, labels=labels, pw_loss=pixel_wise_loss, isTrain=False)
                    else:
                        images = data['images'].to(device)
                        labels = data['labels'].to(device)

                        # anchor is 2nd of last batch
                        images_ = torch.cat((last_two_frames,torch.unsqueeze(images[0,:,:,:],0)),0)
                        labels_ = torch.cat((last_two_labels,torch.unsqueeze(labels[0,:,:,:],0)),0)
                        valLoss += _propagateTriplet(images_, dexi, auto, tc_loss, args, device, labels=labels_, pw_loss=pixel_wise_loss, isTrain=False)

                        # anchor is 3rd of last batch
                        images_ = torch.cat((torch.unsqueeze(last_two_frames[-1,:,:,:],0),images[0:2,:,:,:]),0)
                        labels_ = torch.cat((torch.unsqueeze(last_two_labels[-1,:,:,:],0),labels[0:2,:,:,:]),0)
                        valLoss += _propagateTriplet(images_, dexi, auto, tc_loss, args, device, labels=labels_, pw_loss=pixel_wise_loss, isTrain=False)

                        # anchor is 1st of this batch
                        valLoss += _propagateTriplet(images, dexi, auto, tc_loss, args, device, labels=labels, pw_loss=pixel_wise_loss, isTrain=False)

                        last_two_frames = images[-2:,:,:,:]
                        last_two_labels = labels[-2:,:,:,:]

                elif (j+1)*batch_size < len(val_set):
                    load_val = next(successor_val)
                    num_frames_val += load_val['frames']
                    startVideo = True

        dexi.train()
        
        trainLoss = trainLoss/len(train_loader)
        valLoss   = valLoss/len(val_loader)
        print('Epoch-{} [train-Loss]:{:.4f} [val-Loss]:{:.4f}'.format(epoch+1, trainLoss, valLoss))

        trainlossTracker.append(trainLoss)
        validlossTracker.append(valLoss)

        # Save model params only when validation loss decreases
        if(valLoss<=loss_min):
            torch.save(dexi.state_dict(), SAVE_PATH)
            loss_min=valLoss


# Train DexiNed by Quadruplet loss + MSE/BDCN
def trainQuadruplet(device, args):
    # Parameters
    SAVE_PATH = os.path.join('checkpoints',args.checkpoint)
    IMG_TRAIN_PATH = os.path.join('dataset_origin','train','TC')
    IMG_VAL_PATH = os.path.join('dataset_origin','val','TC')

    LOAD_PATH_DEXI = os.path.join('checkpoints','triplet_mse_w10_best.pth')
    LOAD_PATH_AUTO = os.path.join('checkpoints','BEST_checkpoint.pth')

    # Hyperparameters
    batch_size = 4
    epsilon    = args.epsilon
    epochs     = args.epochs

    # Accelarate training by using cudnn benchmark
    if torch.cuda.is_available():
       torch.backends.cudnn.benchmark=True

    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"

    if not os.path.isfile(LOAD_PATH_DEXI):
       raise FileNotFoundError(
            f"Checkpoint file not found: {LOAD_PATH_DEXI}")
    elif not os.path.isfile(LOAD_PATH_AUTO):
       raise FileNotFoundError(
            f"Checkpoint file not found: {LOAD_PATH_AUTO}")
    print(f"Restoring weights from: {LOAD_PATH_DEXI}")
    print(f"Restoring weights from: {LOAD_PATH_AUTO}")

    dexi = DexiNed().to(device)
    auto = SegNet().to(device)
    dexi.load_state_dict(torch.load(LOAD_PATH_DEXI, map_location=device))
    auto.load_state_dict(torch.load(LOAD_PATH_AUTO, map_location=device))

    auto.eval()

    train_set = datasets.Dataset(IMG_TRAIN_PATH,
                          img_width=args.img_width,
                          img_height=args.img_height,
                          mean_bgr=args.mean_pixel_values[0:3] if len(
                              args.mean_pixel_values) == 4 else args.mean_pixel_values,
                          model=args.model,
                          mode = args.mode,
                          transformer=transform
                          )

    val_set = datasets.Dataset(IMG_VAL_PATH,
                          img_width=args.img_width,
                          img_height=args.img_height,
                          mean_bgr=args.mean_pixel_values[0:3] if len(
                              args.mean_pixel_values) == 4 else args.mean_pixel_values,
                          model=args.model,
                          mode = args.mode,
                          transformer=transform
                          )

    train_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

    optimizer = optim.RMSprop(dexi.parameters(), lr=epsilon, alpha=0.99, eps=1e-08)
    #scheduler = CosineAnnealingWarmRestarts(optimizer,T_0 = 1,T_mult=2,eta_min=0.000000001)

    tc_loss = QuadrupletLoss(margin1=args.margin1, margin2=args.margin2)
    
    if args.ploss == 'bdcn':
        pixel_wise_loss = bdcn_loss2
    elif args.ploss == 'mse':
        pixel_wise_loss = nn.MSELoss()

    # Define a scale tensor replicates mean_pixel_vaule with same dim of data
    norm_tensor = torch.zeros(3, args.img_width, args.img_height)
    norm_tensor[0] += args.mean_pixel_values[0]
    norm_tensor[1] += args.mean_pixel_values[1]
    norm_tensor[2] += args.mean_pixel_values[2]
    norm_tensor = norm_tensor.repeat(batch_size,1,1,1).to(device)

    trainlossTracker = []
    validlossTracker = []
    loss_min   = 999999

    print('Started training DexiNed with Quadruplet metric...')
    for epoch in range(epochs):
        trainLoss = 0
        valLoss   = 0
        successor = iter(train_set.video_frames)
        load = next(successor)
        num_frames = load['frames']
        startVideo = True
        last_three_frames = None
        last_three_labels = None
        for i, data in enumerate(train_loader):
            
            if (i+1)*batch_size <= num_frames:
                if startVideo:
                    startVideo = False
                    # Feed data into DexiNed
                    images = data['images'].to(device)
                    labels = data['labels'].to(device)
                    last_three_frames = images[-3:,:,:,:]
                    last_three_labels = labels[-3:,:,:,:]

                    trainLoss += _propagateQuadruplet(images, dexi, auto, tc_loss, args, device, labels=labels, pw_loss=pixel_wise_loss, opt=optimizer)
                else:
                    images = data['images'].to(device)
                    labels = data['labels'].to(device)

                    # anchor is 2nd of last batch
                    images_ = torch.cat((last_three_frames,torch.unsqueeze(images[0,:,:,:],0)),0)
                    labels_ = torch.cat((last_three_labels,torch.unsqueeze(labels[0,:,:,:],0)),0)
                    trainLoss += _propagateQuadruplet(images_, dexi, auto, tc_loss, args, device, labels=labels_, pw_loss=pixel_wise_loss, opt=optimizer)

                    # anchor is 3rd of last batch
                    images_ = torch.cat((last_three_frames[-2:,:,:,:],images[0:2,:,:,:]),0)
                    labels_ = torch.cat((last_three_labels[-2:,:,:,:],labels[0:2,:,:,:]),0)
                    trainLoss += _propagateQuadruplet(images_, dexi, auto, tc_loss, args, device, labels=labels_, pw_loss=pixel_wise_loss, opt=optimizer)

                    # anchor is 4th of last batch
                    images_ = torch.cat((torch.unsqueeze(last_three_frames[-1,:,:,:],0),images[0:3,:,:,:]),0)
                    labels_ = torch.cat((torch.unsqueeze(last_three_labels[-1,:,:,:],0),labels[0:3,:,:,:]),0)
                    trainLoss += _propagateQuadruplet(images_, dexi, auto, tc_loss, args, device, labels=labels_, pw_loss=pixel_wise_loss, opt=optimizer)

                    # anchor is 1st of this batch
                    trainLoss += _propagateQuadruplet(images, dexi, auto, tc_loss, args, device, labels=labels, pw_loss=pixel_wise_loss, opt=optimizer)

                    last_three_frames = images[-3:,:,:,:]
                    last_three_labels = labels[-3:,:,:,:]

            elif (i+1)*batch_size < len(train_set):
                load = next(successor)
                num_frames += load['frames']
                startVideo = True

        
        # Validation
        dexi.eval()
        with torch.no_grad():
            successor_val = iter(val_set.video_frames)
            load_val = next(successor_val)
            num_frames_val = load_val['frames']
            startVideo = True
            last_three_frames = None
            last_three_labels = None
            for j, data in enumerate(val_loader):

                if (j+1)*batch_size <= num_frames_val:
                    if startVideo:
                        startVideo = False
                        # Feed data into DexiNed
                        images = data['images'].to(device)
                        labels = data['labels'].to(device)
                        last_three_frames = images[-3:,:,:,:]
                        last_three_labels = labels[-3:,:,:,:]

                        valLoss += _propagateQuadruplet(images, dexi, auto, tc_loss, args, device, labels=labels, pw_loss=pixel_wise_loss, isTrain=False)
                    else:
                        images = data['images'].to(device)
                        labels = data['labels'].to(device)

                        # anchor is 2nd of last batch
                        images_ = torch.cat((last_three_frames,torch.unsqueeze(images[0,:,:,:],0)),0)
                        labels_ = torch.cat((last_three_labels,torch.unsqueeze(labels[0,:,:,:],0)),0)
                        trainLoss += _propagateQuadruplet(images_, dexi, auto, tc_loss, args, device, labels=labels, pw_loss=pixel_wise_loss, isTrain=False)

                        # anchor is 3rd of last batch
                        images_ = torch.cat((last_three_frames[-2:,:,:,:],images[0:2,:,:,:]),0)
                        labels_ = torch.cat((last_three_labels[-2:,:,:,:],labels[0:2,:,:,:]),0)
                        trainLoss += _propagateQuadruplet(images_, dexi, auto, tc_loss, args, device, labels=labels, pw_loss=pixel_wise_loss, isTrain=False)

                        # anchor is 4th of last batch
                        images_ = torch.cat((torch.unsqueeze(last_three_frames[-1,:,:,:],0),images[0:3,:,:,:]),0)
                        labels_ = torch.cat((torch.unsqueeze(last_three_labels[-1,:,:,:],0),labels[0:3,:,:,:]),0)
                        trainLoss += _propagateQuadruplet(images_, dexi, auto, tc_loss, args, device, labels=labels, pw_loss=pixel_wise_loss, isTrain=False)

                        # anchor is 1st of this batch
                        trainLoss += _propagateQuadruplet(images, dexi, auto, tc_loss, args, device, labels=labels, pw_loss=pixel_wise_loss, isTrain=False)

                        last_three_frames = images[-3:,:,:,:]
                        last_three_labels = labels[-3:,:,:,:]

                elif (j+1)*batch_size < len(val_set):
                    load_val = next(successor_val)
                    num_frames_val += load_val['frames']
                    startVideo = True

        dexi.train()
        
        trainLoss = trainLoss/len(train_loader)
        valLoss   = valLoss/len(val_loader)
        print('Epoch-{} [train-Loss]:{:.4f} [val-Loss]:{:.4f}'.format(epoch+1, trainLoss, valLoss))

        trainlossTracker.append(trainLoss)
        validlossTracker.append(valLoss)

        # Save model params only when validation loss decreases
        if(valLoss<=loss_min):
            torch.save(dexi.state_dict(), SAVE_PATH)
            loss_min=valLoss


''' network propagation functions '''
def _propagateTriplet(batch, model1, model2, consistency, args, device, labels=None, pw_loss=None, opt=None, isTrain=True):
    output_list = model1(batch)
    
    output = output_list[-1]
    output = torch.sigmoid(output)
    rescaled = 1.0 - output
    output = rescaled.repeat(1,3,1,1)+0.1

    # Feed DexiNed prediction into SegNet
    encode, _ = model2(output)
    encode = torch.flatten(encode,start_dim=1)
    
    # Compute loss
    if labels == None:
        loss = consistency(encode[0],encode[1],encode[2])
    else:
        if args.ploss == 'bdcn':
            l_weight = [0.7,0.7,1.1,1.1,0.3,0.3,1.3] # New BDCN loss

            output_list = [torch.unsqueeze(preds[0,:,:,:],0) for preds in output_list]
            loss_ = sum([pw_loss(preds, torch.unsqueeze(torch.unsqueeze(labels[0,0,:,:],0),0),l_w) for preds, l_w in zip(output_list,l_weight)]) # bdcn_loss
        elif args.ploss == 'mse':
            loss_ = pw_loss(output[0], labels[0])
        
        #loss = consistency(encode[0],encode[1],encode[2]) + weight*loss_
        loss = (1-args.weight)*consistency(encode[0],encode[1],encode[2]) + args.weight*loss_

    if isTrain:
        loss.backward()
        opt.step()

    return loss.item()

def _propagateQuadruplet(batch, model1, model2, consistency, args, device, labels=None, pw_loss=None, opt=None, isTrain=True):
    output_list = model1(batch)
    
    output = output_list[-1]
    output = torch.sigmoid(output)
    rescaled = 1.0 - output
    output = rescaled.repeat(1,3,1,1)+0.1

    # Feed DexiNed prediction into SegNet
    encode, _ = model2(output)
    encode = torch.flatten(encode,start_dim=1)
    
    # Compute loss
    if labels == None:
        loss = consistency(encode[0],encode[1],encode[2],encode[3])
    else:
        if args.ploss == 'bdcn':
            l_weight = [0.7,0.7,1.1,1.1,0.3,0.3,1.3] # New BDCN loss
        
            output_list = [torch.unsqueeze(preds[0,:,:,:],0) for preds in output_list]
            loss_ = sum([pw_loss(preds, torch.unsqueeze(torch.unsqueeze(labels[0,0,:,:],0),0),l_w) for preds, l_w in zip(output_list,l_weight)]) # bdcn_loss
        elif args.ploss == 'mse':
            loss_ = pw_loss(output[0], labels[0])
        
        #loss = consistency(encode[0],encode[1],encode[2],encode[3]) + weight*loss_
        loss = (1-args.weight)*consistency(encode[0],encode[1],encode[2],encode[3]) + args.weight*loss_

    if isTrain:
        loss.backward()
        opt.step()

    return loss.item()