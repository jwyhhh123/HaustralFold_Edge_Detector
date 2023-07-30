import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics.classification import MulticlassJaccardIndex

import datasets
from utils import metric
from utils import warp
from utils import preprocess
from utils.metric import distanceTransform
from third_party.Autoencoder.models import SegNet
from third_party.DexiNed.models import DexiNed
from third_party.FlowNet2.models import FlowNet2
from third_party.DexiNed.utils.image import save_image_batch_to_disk


transform = transforms.Compose([
    transforms.GaussianBlur(3)
])


# Binarise input with a adjustable threshold
def thresholding(img, threshold=240):
    img[img>threshold]=255
    img[img<=threshold]=0
    return img


# Test SegNet [deprecated]
def testSegNet(device, args):
    # Parameters
    IMG_VAL_PATH = args.data_path
    LOAD_PATH_DEXI = os.path.join('checkpoints','10_model.pth')
    LOAD_PATH_AUTO = os.path.join('checkpoints','BEST_checkpoint.pth')

    # Hyperparameters
    batch_size = 1

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

    dexi.eval()
    auto.eval()

    val_set = datasets.Dataset(IMG_VAL_PATH,
                          img_width=args.img_width,
                          img_height=args.img_height,
                          mean_bgr=args.mean_pixel_values[0:3] if len(
                              args.mean_pixel_values) == 4 else args.mean_pixel_values,
                          model=args.model,
                          mode = args.mode,
                          transformer=transform
                          )

    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

    criterion = nn.MSELoss()

    print('> Start test DexiNed and save prediction masks')
    successor = iter(dataset.video_frames)
    load = next(successor)
    with torch.no_grad():
        total_duration = []
        num_frames = load['frames']
        for idx, data in enumerate(test_loader):
            images = data['images'].to(device)
            file_names = data['file_names']
            image_shape = data['image_shape']
            #print(f"input tensor shape: {images.shape}")

            # images = images[:, [2, 1, 0], :, :]
            start_time = time.time()
            preds = dexi(images)

            output = output_list[-1]
            output = torch.sigmoid(output)
            rescaled = 1.0 - output
            output = rescaled.repeat(1,3,1,1)

            # Feed DexiNed prediction into SegNet
            _, decode = auto(output)
            out = decode.detach().cpu().numpy()
            out = np.transpose(out, (1, 2, 0))
            cv2.imshow('input', output[0,0,:,:].detach().cpu().numpy())
            cv2.imshow('decoded', mask)
            cv2.waitKey()


# Test trained DexiNed and save edge predictions
def testDexiNed(device, args):
    # Parameters
    LOAD_PATH = os.path.join('checkpoints', args.checkpoint)
    IMG_VAL_PATH  = args.data_path
    SAVE_PATH = args.result_path

    # Hyperparameters
    batch_size = 1

    if not os.path.isfile(LOAD_PATH):
       raise FileNotFoundError(
            f"Checkpoint filte note found: {LOAD_PATH}")
    print(f"Restoring weights from: {LOAD_PATH}")

    model = DexiNed().to(device)
    model.load_state_dict(torch.load(LOAD_PATH, map_location=device))
    model.eval()

    dataset = datasets.Dataset(IMG_VAL_PATH,
                          img_width=args.img_width,
                          img_height=args.img_height,
                          mean_bgr=args.mean_pixel_values[0:3] if len(
                              args.mean_pixel_values) == 4 else args.mean_pixel_values,
                          model= args.model,
                          mode = args.mode,
                          transformer=transform
                          )
    test_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

    # Define custom mask
    mask_h = preprocess.findHyperKvasirMask(args.img_height,args.img_width,itr=6)
    mask_h = mask_h.transpose(2, 0, 1)[0]

    print('> Start test DexiNed and save prediction masks')
    successor = iter(dataset.video_frames)
    load = next(successor)
    with torch.no_grad():
        total_duration = []
        num_frames = load['frames']
        for idx, data in enumerate(test_loader):
            images = data['images'].to(device)
            file_names = data['file_names']
            image_shape = data['image_shape']
            #print(f"input tensor shape: {images.shape}")

            # images = images[:, [2, 1, 0], :, :]
            start_time = time.time()
            preds = model(images)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            
            #print((idx+1),'/',num_frames)

            if (idx+1) <= num_frames:
                path = os.path.join(SAVE_PATH, load['snippet'])
                #print('Saving result to: ', path)
                save_image_batch_to_disk(preds,
                                        path,
                                        file_names,
                                        image_shape,
                                        arg=args,
                                        mask = mask_h,
                                        single_channel = True)
            else:
                load = next(successor)
                num_frames += load['frames']
                path = os.path.join(SAVE_PATH, load['snippet'])
                #print('Saving result to: ', path)
                save_image_batch_to_disk(preds,
                                        path,
                                        file_names,
                                        image_shape,
                                        arg=args,
                                        mask = mask_h,
                                        single_channel = True)
            torch.cuda.empty_cache()

    total_duration = np.array(total_duration)
    print("******** Testing finished *******")
    print("Average time per image: %f.4" % total_duration.mean(), "seconds")
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")


# Evaluate TC on edge maps predicted by [CHECK_POINT]
def evalConsistency(device, args):
    # Parameters
    IMG_VAL_PATH = os.path.join('dataset_origin','test')

    LOAD_PATH_DEXI = os.path.join('checkpoints', args.checkpoint)
    LOAD_PATH_FLOW = os.path.join('checkpoints','FlowNet2_checkpoint.pth.tar')

    # Hyperparameters
    batch_size = 2

    # Accelarate training by using cudnn benchmark
    if torch.cuda.is_available():
       torch.backends.cudnn.benchmark=True

    if not os.path.isfile(LOAD_PATH_DEXI):
       raise FileNotFoundError(
            f"Checkpoint file not found: {LOAD_PATH_DEXI}")
    elif not os.path.isfile(LOAD_PATH_FLOW):
       raise FileNotFoundError(
            f"Checkpoint file not found: {LOAD_PATH_FLOW}")
    print(f"Restoring weights from: {LOAD_PATH_DEXI}")
    print(f"Restoring weights from: {LOAD_PATH_FLOW}")

    dexi = DexiNed().to(device)
    fn   = FlowNet2(args).to(device)

    dexi.load_state_dict(torch.load(LOAD_PATH_DEXI, map_location=device))
    fn.load_state_dict(torch.load(LOAD_PATH_FLOW)["state_dict"])

    dexi.eval()
    fn.eval()

    val_set = datasets.Dataset(IMG_VAL_PATH,
                          img_width=args.img_width,
                          img_height=args.img_height,
                          mean_bgr=args.mean_pixel_values[0:3] if len(
                              args.mean_pixel_values) == 4 else args.mean_pixel_values,
                          model= args.model,
                          mode = 'test', # test mode is kept
                          transformer=transform
                          )

    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2)

    # weighted IoU score
    jaccard = MulticlassJaccardIndex(num_classes=2,average='weighted').to(device)

    # Find mask for hyperkvasir data
    mask_h = preprocess.findHyperKvasirMask(args.img_height,args.img_width,itr=6)
    mask_h = mask_h.transpose(2, 0, 1)
    mask_h = torch.from_numpy(mask_h.copy()).float().to(device)
    
    print('> Start consistency evaluation of ', args.checkpoint)
    #chamfer_score = []
    iou_score = []
    avg_edge_videos = []
    with torch.no_grad():
        #valLoss = 0 #placeholder, for evaluate purpose
        successor = iter(val_set.video_frames)
        load = next(successor)
        num_frames = load['frames']
        startVideo = True
        last_frame = None
        last_preds = None
        #metric1 = 0
        metric2 = 0
        avg_edge_pixels = 0
        for i, data in enumerate(val_loader):
            fstFrame = None
            scdFrame = None
            if (i+1)*batch_size <= num_frames:
                if startVideo:
                    startVideo = False
                    
                    images = data['images'].to(device)
                    output = dexi(images)[-1]
                    output = torch.sigmoid(output)
                    rescaled = 1.0 - output
                    preds = rescaled.repeat(1,3,1,1)*255

                    preds = thresholding(preds, threshold=args.thr1)

                    avg_edge_pixels += (preds[:,0,:,:]==0).sum()
                    
                    last_frame = images[-1,:,:,:]
                    last_preds = preds[-1,:,:,:]

                    # compute optical flow
                    im = torch.swapaxes(images, 0, 1).unsqueeze(0).to(device)
                    flow = fn(im).squeeze()
                    flow = flow.data.cpu().numpy().transpose(1, 2, 0)

                    # Apply mask to remove intensity at the border
                    fstFrame = preds[0]+mask_h
                    scdFrame = preds[1]+mask_h
                    fstFrame[fstFrame>255]=255
                    scdFrame[scdFrame>255]=255
                    
                    # Find warped edge
                    warped = warp.findWarped(fstFrame,scdFrame,flow, isTorch=True).to(device)

                    fstFrame_ = distanceTransform(fstFrame.cpu(), isTensor=True).to(device)
                    scdFrame_ = distanceTransform(scdFrame.cpu(), isTensor=True).to(device)
                    warped_ = distanceTransform(warped.cpu(), isTensor=True).to(device)

                    #fstFrame_ = fstFrame_.repeat(3,1,1)
                    #scdFrame_ = scdFrame_.repeat(3,1,1)
                    #warped_ = warped_.repeat(3,1,1)

                    # Thresholding
                    #fstFrame = (thresholding(fstFrame)/255).int()
                    #scdFrame = (thresholding(scdFrame)/255).int()
                    #warped = (thresholding(warped)/255).int()

                    fstFrame_ = (thresholding(fstFrame_, threshold=args.thr2)/255).int()
                    scdFrame_ = (thresholding(scdFrame_, threshold=args.thr2)/255).int()
                    warped_ = (thresholding(warped_, threshold=args.thr2)/255).int()

                    ## Evaluate local consistency at consecutive frames
                    # local chamfer score
                    #loss01 = 1-metric.chamferDistance(fstFrame,scdFrame)
                    #loss02 = 1-metric.chamferDistance(scdFrame,warped)

                    # local IoU
                    #iou01 = jaccard(fstFrame_,scdFrame_)
                    iou02 = jaccard(scdFrame_,warped_)

                    #metric1 += loss02
                    metric2 += iou02.cpu().numpy()

                else:
                    images = data['images'].to(device)
                    output = dexi(images)[-1]
                    output = torch.sigmoid(output)
                    rescaled = 1.0 - output
                    preds = rescaled.repeat(1,3,1,1)*255

                    preds = thresholding(preds,threshold=args.thr1)

                    avg_edge_pixels += (preds[:,0,:,:]==0).sum()

                    ## feature matching for last and current first frames
                    images_ = torch.cat((torch.unsqueeze(last_frame,0),torch.unsqueeze(images[0,:,:,:],0)),0)
                    preds_  = torch.cat((torch.unsqueeze(last_preds,0),torch.unsqueeze(preds[0,:,:,:],0)),0)
                    
                    # compute optical flow
                    im = torch.swapaxes(images_, 0, 1).unsqueeze(0).to(device)
                    flow = fn(im).squeeze()
                    flow = flow.data.cpu().numpy().transpose(1, 2, 0)

                    # Apply mask to remove intensity at the border
                    fstFrame = preds_[0]+mask_h
                    scdFrame = preds_[1]+mask_h
                    fstFrame[fstFrame>255]=255
                    scdFrame[scdFrame>255]=255

                    # Find warped edge
                    warped = warp.findWarped(fstFrame,scdFrame,flow, isTorch=True).to(device)

                    fstFrame_ = distanceTransform(fstFrame.cpu(), isTensor=True).to(device)
                    scdFrame_ = distanceTransform(scdFrame.cpu(), isTensor=True).to(device)
                    warped_ = distanceTransform(warped.cpu(), isTensor=True).to(device)

                    #fstFrame_ = fstFrame_.repeat(3,1,1)
                    #scdFrame_ = scdFrame_.repeat(3,1,1)
                    #warped_ = warped_.repeat(3,1,1)

                    # Thresholding
                    #fstFrame = (thresholding(fstFrame)/255).int()
                    #scdFrame = (thresholding(scdFrame)/255).int()
                    #warped = (thresholding(warped)/255).int()

                    fstFrame_ = (thresholding(fstFrame_, threshold=args.thr2)/255).int()
                    scdFrame_ = (thresholding(scdFrame_, threshold=args.thr2)/255).int()
                    warped_ = (thresholding(warped_, threshold=args.thr2)/255).int()

                    ## Evaluate local consistency at consecutive frames
                    # local chamfer score
                    #loss01 = 1-metric.chamferDistance(fstFrame,scdFrame)
                    #loss02 = 1-metric.chamferDistance(scdFrame,warped)

                    # local IoU
                    #iou01 = jaccard(fstFrame_,scdFrame_)
                    iou02 = jaccard(scdFrame_,warped_)

                    #metric1 += loss02
                    metric2 += iou02.cpu().numpy()

                    ## match again for current two frames
                    # compute optical flow
                    im = torch.swapaxes(images, 0, 1).unsqueeze(0).to(device)
                    flow = fn(im).squeeze()
                    flow = flow.data.cpu().numpy().transpose(1, 2, 0)

                    # Apply mask to remove intensity at the border
                    fstFrame = preds[0]+mask_h
                    scdFrame = preds[1]+mask_h
                    fstFrame[fstFrame>255]=255
                    scdFrame[scdFrame>255]=255

                    # Find warped edge
                    warped = warp.findWarped(fstFrame,scdFrame,flow, isTorch=True).to(device)

                    fstFrame_ = distanceTransform(fstFrame.cpu(), isTensor=True).to(device)
                    scdFrame_ = distanceTransform(scdFrame.cpu(), isTensor=True).to(device)
                    warped_ = distanceTransform(warped.cpu(), isTensor=True).to(device)

                    #fstFrame_ = fstFrame_.repeat(3,1,1)
                    #scdFrame_ = scdFrame_.repeat(3,1,1)
                    #warped_ = warped_.repeat(3,1,1)

                    # Thresholding
                    #fstFrame = (thresholding(fstFrame)/255).int()
                    #scdFrame = (thresholding(scdFrame)/255).int()
                    #warped = (thresholding(warped)/255).int()

                    fstFrame_ = (thresholding(fstFrame_, threshold=args.thr2)/255).int()
                    scdFrame_ = (thresholding(scdFrame_, threshold=args.thr2)/255).int()
                    warped_ = (thresholding(warped_, threshold=args.thr2)/255).int()

                    ## Evaluate local consistency at consecutive frames
                    # local chamfer score
                    #loss01 = 1-metric.chamferDistance(fstFrame,scdFrame)
                    #loss02 = 1-metric.chamferDistance(scdFrame,warped)

                    # local IoU
                    #iou01 = jaccard(fstFrame_,scdFrame_)
                    iou02 = jaccard(scdFrame_,warped_)

                    #metric1 += loss02
                    metric2 += iou02.cpu().numpy()

                    last_frame = images[-1,:,:,:]
                    last_preds = preds[-1,:,:,:]

            elif (i+1)*batch_size < len(val_set):
                #metric1 = metric1/(load['frames']-1)
                metric2 = metric2/(load['frames']-1)
                avg_edge_pixels = avg_edge_pixels/(load['frames']-1)
                #chamfer_score.append(metric1)
                iou_score.append(metric2)
                avg_edge_videos.append(avg_edge_pixels.detach().cpu().numpy())

                # log
                #print('current video with chamfer score: {}'.format(metric1))
                print('current video with iou score: {}'.format(metric2))

                load = next(successor)
                num_frames += load['frames']
                startVideo = True
                #metric1 = 0
                metric2 = 0

    # Calculate mean and deviation
    #mean_chamfer = np.mean(chamfer_score)
    mean_iou = np.mean(iou_score)
    #std_chamfer = np.std(chamfer_score)
    std_iou = np.std(iou_score)

    # Calculate mean percentage of pixels in an output
    mean_edge = np.mean(avg_edge_videos)/(args.img_width*args.img_height)

    #print('Computed mean of chamfer score {} with standard deviation {}'.format(mean_chamfer, std_chamfer))
    print('Computed mean of IoU {} with standard deviation {}'.format(mean_iou, std_iou))
    print('Computed mean of edge pixel rate: {}'.format(mean_edge))


# Evaluate TC on edge maps predicted by Foldit
def evalConsistencyFoldit(device, args):
    # Parameters
    IMG_VAL_PATH = os.path.join('dataset_origin','foldit','TC')
    LOAD_PATH_FLOW = os.path.join('checkpoints','FlowNet2_checkpoint.pth.tar')

    # Hyperparameters
    batch_size = 2

    # Accelarate training by using cudnn benchmark
    if torch.cuda.is_available():
       torch.backends.cudnn.benchmark=True

    if not os.path.isfile(LOAD_PATH_FLOW):
       raise FileNotFoundError(
            f"Checkpoint file not found: {LOAD_PATH_FLOW}")
    print(f"Restoring weights from: {LOAD_PATH_FLOW}")

    fn   = FlowNet2(args).to(device)
    fn.load_state_dict(torch.load(LOAD_PATH_FLOW)["state_dict"])

    fn.eval()

    val_set = datasets.Dataset(IMG_VAL_PATH,
                          img_width=args.img_width,
                          img_height=args.img_height,
                          mean_bgr=args.mean_pixel_values[0:3] if len(
                              args.mean_pixel_values) == 4 else args.mean_pixel_values,
                          model= args.model,
                          mode = 'train', # test mode is kept
                          transformer=transform
                          )

    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2)

    # weighted IoU score
    jaccard = MulticlassJaccardIndex(num_classes=2,average='weighted').to(device)

    # Find mask for hyperkvasir data
    mask_h = preprocess.findHyperKvasirMask(args.img_height,args.img_width,itr=6)
    mask_h = mask_h.transpose(2, 0, 1)
    mask_h = torch.from_numpy(mask_h.copy()).float().to(device)
    
    print('> Start consistency evaluation of Foldit')
    #chamfer_score = []
    iou_score = []
    avg_edge_videos = []
    with torch.no_grad():
        #valLoss = 0 #placeholder, for evaluate purpose
        successor = iter(val_set.video_frames)
        load = next(successor)
        num_frames = load['frames']
        startVideo = True
        last_frame = None
        last_preds = None
        #metric1 = 0
        metric2 = 0
        avg_edge_pixels = 0
        for i, data in enumerate(val_loader):
            fstFrame = None
            scdFrame = None
            if (i+1)*batch_size <= num_frames:
                if startVideo:
                    startVideo = False
                    
                    images = data['images'].to(device)
                    labels = data['labels'].to(device)
                    preds = labels*255

                    preds = thresholding(preds,threshold=50)
                    
                    avg_edge_pixels += (preds[:,0,:,:]==0).sum()
                    
                    last_frame = images[-1,:,:,:]
                    last_preds = preds[-1,:,:,:]

                    # compute optical flow
                    im = torch.swapaxes(images, 0, 1).unsqueeze(0).to(device)
                    flow = fn(im).squeeze()
                    flow = flow.data.cpu().numpy().transpose(1, 2, 0)

                    # Apply mask to remove intensity at the border
                    fstFrame = preds[0]+mask_h
                    scdFrame = preds[1]+mask_h
                    fstFrame[fstFrame>255]=255
                    scdFrame[scdFrame>255]=255
                    
                    # Find warped edge
                    warped = warp.findWarped(fstFrame,scdFrame,flow, isTorch=True).to(device)

                    fstFrame_ = distanceTransform(fstFrame.cpu(), isTensor=True).to(device)
                    scdFrame_ = distanceTransform(scdFrame.cpu(), isTensor=True).to(device)
                    warped_ = distanceTransform(warped.cpu(), isTensor=True).to(device)

                    fstFrame_ = (thresholding(fstFrame_, threshold=args.thr2)/255).int()
                    scdFrame_ = (thresholding(scdFrame_, threshold=args.thr2)/255).int()
                    warped_ = (thresholding(warped_, threshold=args.thr2)/255).int()

                    # local IoU
                    #iou01 = jaccard(fstFrame_,scdFrame_)
                    iou02 = jaccard(scdFrame_,warped_)

                    #metric1 += loss02
                    metric2 += iou02.cpu().numpy()

                else:
                    images = data['images'].to(device)
                    labels = data['labels'].to(device)
                    preds = labels*255

                    preds = thresholding(preds,threshold=50)

                    avg_edge_pixels += (preds[:,0,:,:]==0).sum()

                    ## feature matching for last and current first frames
                    images_ = torch.cat((torch.unsqueeze(last_frame,0),torch.unsqueeze(images[0,:,:,:],0)),0)
                    preds_  = torch.cat((torch.unsqueeze(last_preds,0),torch.unsqueeze(preds[0,:,:,:],0)),0)
                    
                    # compute optical flow
                    im = torch.swapaxes(images_, 0, 1).unsqueeze(0).to(device)
                    flow = fn(im).squeeze()
                    flow = flow.data.cpu().numpy().transpose(1, 2, 0)

                    # Apply mask to remove intensity at the border
                    fstFrame = preds_[0]+mask_h
                    scdFrame = preds_[1]+mask_h
                    fstFrame[fstFrame>255]=255
                    scdFrame[scdFrame>255]=255

                    # Find warped edge
                    warped = warp.findWarped(fstFrame,scdFrame,flow, isTorch=True).to(device)

                    fstFrame_ = distanceTransform(fstFrame.cpu(), isTensor=True).to(device)
                    scdFrame_ = distanceTransform(scdFrame.cpu(), isTensor=True).to(device)
                    warped_ = distanceTransform(warped.cpu(), isTensor=True).to(device)

                    fstFrame_ = (thresholding(fstFrame_, threshold=args.thr2)/255).int()
                    scdFrame_ = (thresholding(scdFrame_, threshold=args.thr2)/255).int()
                    warped_ = (thresholding(warped_, threshold=args.thr2)/255).int()

                    # local IoU
                    #iou01 = jaccard(fstFrame_,scdFrame_)
                    iou02 = jaccard(scdFrame_,warped_)

                    #metric1 += loss02
                    metric2 += iou02.cpu().numpy()

                    ## match again for current two frames
                    # compute optical flow
                    im = torch.swapaxes(images, 0, 1).unsqueeze(0).to(device)
                    flow = fn(im).squeeze()
                    flow = flow.data.cpu().numpy().transpose(1, 2, 0)

                    # Apply mask to remove intensity at the border
                    fstFrame = preds[0]+mask_h
                    scdFrame = preds[1]+mask_h
                    fstFrame[fstFrame>255]=255
                    scdFrame[scdFrame>255]=255

                    # Find warped edge
                    warped = warp.findWarped(fstFrame,scdFrame,flow, isTorch=True).to(device)

                    fstFrame_ = distanceTransform(fstFrame.cpu(), isTensor=True).to(device)
                    scdFrame_ = distanceTransform(scdFrame.cpu(), isTensor=True).to(device)
                    warped_ = distanceTransform(warped.cpu(), isTensor=True).to(device)

                    fstFrame_ = (thresholding(fstFrame_, threshold=args.thr2)/255).int()
                    scdFrame_ = (thresholding(scdFrame_, threshold=args.thr2)/255).int()
                    warped_ = (thresholding(warped_, threshold=args.thr2)/255).int()

                    # local IoU
                    #iou01 = jaccard(fstFrame_,scdFrame_)
                    iou02 = jaccard(scdFrame_,warped_)

                    #metric1 += loss02
                    metric2 += iou02.cpu().numpy()

                    last_frame = images[-1,:,:,:]
                    last_preds = preds[-1,:,:,:]

            elif (i+1)*batch_size < len(val_set):
                #metric1 = metric1/(load['frames']-1)
                metric2 = metric2/(load['frames']-1)
                avg_edge_pixels = avg_edge_pixels/(load['frames']-1)
                #chamfer_score.append(metric1)
                iou_score.append(metric2)
                avg_edge_videos.append(avg_edge_pixels.detach().cpu().numpy())

                # log
                #print('current video with chamfer score: {}'.format(metric1))
                print('current video with iou score: {}'.format(metric2))

                load = next(successor)
                num_frames += load['frames']
                startVideo = True
                #metric1 = 0
                metric2 = 0

    # Calculate mean and deviation
    #mean_chamfer = np.mean(chamfer_score)
    mean_iou = np.mean(iou_score)
    #std_chamfer = np.std(chamfer_score)
    std_iou = np.std(iou_score)

    # Calculate mean percentage of pixels in an output
    mean_edge = np.mean(avg_edge_videos)/(args.img_width*args.img_height)

    #print('Computed mean of chamfer score {} with standard deviation {}'.format(mean_chamfer, std_chamfer))
    print('Computed mean of IoU {} with standard deviation {}'.format(mean_iou, std_iou))
    print('Computed mean of edge pixel rate: {}'.format(mean_edge))