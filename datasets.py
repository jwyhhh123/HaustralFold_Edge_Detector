import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.preprocess import findHyperKvasirMask, findSunMask


class Dataset(Dataset):
    def __init__(self,
                 data_root,
                 mean_bgr,
                 img_height,
                 img_width,
                 model,
                 mode,
                 transformer = None
                 ):

        self.data_root = data_root
        self.gt_root = os.path.join(data_root[:-3],'gt')
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.model = model
        self.mode = mode
        self.tf = transformer
        self.video_frames = []
        self.data_index = self._build_index()
        self.mask_hyperkvasir = findHyperKvasirMask(self.img_height,self.img_width).astype(float)
        #self.mask_sun = findSunMask(self.img_height,self.img_width).astype(float)

        if self.mode == 'track':
            gt_root = os.path.join('result','test')

    def _build_index(self):
        sample_indices = []

        dir_path = sorted(os.listdir(self.data_root), key=len)
        if self.model == 'dexi' or self.model == '':
            dir_path.remove('videos')
        
        # load image paths
        images_path = []
        for d in dir_path:
            dp = os.path.join(self.data_root, d)
            fp = sorted(os.listdir(dp), key=len) # keep the order of video frames
            #print(fp)
            dp = dp + os.sep
            full_paths = [dp + f for f in fp]
            self.video_frames.append(dict(snippet = d, frames = len(full_paths)))
            images_path.extend(full_paths)
        #print(images_path)
        
        # load label paths if they exist
        if self.mode == 'test':
            labels_path = None
        else:
            dir_path = sorted(os.listdir(self.gt_root), key=len)
            
            labels_path = []
            for d in dir_path:
                dp = os.path.join(self.gt_root, d)
                fp = sorted(os.listdir(dp), key=len)
                dp = dp + os.sep
                full_paths = [dp + f for f in fp]
                labels_path.extend(full_paths)

        sample_indices = [images_path, labels_path]
        return sample_indices

    def __len__(self):
        return len(self.data_index[0])

    def __getitem__(self, idx):
        # get data sample
        image_path = self.data_index[0][idx]

        file_name = os.path.basename(image_path)
        file_name = file_name.split('.')[0] + '.png'

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gt = None

        # load pseudo-ground truth
        if not (self.mode == 'test'):
            label_path = self.data_index[1][idx]
            gt = cv2.imread(label_path, cv2.IMREAD_COLOR)

        image, gt, loc = self.transform(image,gt,image_path) # Add label arg here,
        if self.tf:
            image = self.tf(image)

        im_shape = [image.shape[1], image.shape[2]]

        return dict(images=image, labels=gt, file_names=file_name, location=loc, image_shape=im_shape)

    def transform(self, img, gt, image_path):
        img_height = self.img_height
        img_width = self.img_width
        
        img = cv2.resize(img, (img_width,img_height))
        img = np.array(img, dtype=np.float32)

        # Apply masks to remove intensity at the boundary
        if 'subset' in image_path:
            img = img - self.mask_hyperkvasir
            loc = 'hyp'
        else:
            loc = 'n/a'
        
        img[img<0]=0

        # To tensor array and data transform
        if self.model == 'dexi':
            img -= self.mean_bgr
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img.copy()).float()
        elif self.model == 'segnet':
            img = img.transpose(2, 0, 1)
            img = torch.FloatTensor(img / 255.)
        else:
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img.copy()).float()

        # Transform grouth truth labels if they exist
        if self.mode == 'test':
            gt = np.zeros((img_width,img_height))
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            gt = cv2.resize(gt,(img_width,img_height))
            gt = np.array(gt, dtype=np.float32)

            # Apply masks to remove intensity at the boundary
            if 'subset' in image_path:
                gt = gt + self.mask_hyperkvasir
                gt[gt>255]=255
            
            gt[gt<0]=0

            gt = gt.transpose((2, 0, 1))
            gt = gt/255.
            gt = torch.from_numpy(gt.copy()).float()

        return img,gt,loc