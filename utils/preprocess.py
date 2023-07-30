import os
import cv2
import shutil
import numpy as np

''' Video frame extractor '''

def extractImages(pathIn, pathOut):

    '''
    if not os.path.exists(pathIn):
       raise FileNotFoundError(
            f"Video directory not found: {pathIn}")
    '''

    if not os.path.exists(pathIn):
        print("Video directory not found: ", pathIn)
    else:
        print("Processing video path: ", pathIn)
        filesin = next(os.walk(pathIn), (None, None, []))[2]
        filesout = os.listdir(pathOut)
        #print(filesout)

        idx = 0
        for file in filesin:
            count = 0
            dir_name = 'subset' + str(idx)
            pathWrite = os.path.join(pathOut, dir_name)
            vidcap = cv2.VideoCapture(os.path.join(pathIn, file))
            success,image = vidcap.read()
            success = True

            if dir_name in filesout:
                shutil.rmtree(pathWrite)
            os.mkdir(pathWrite)

            while success:
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*150))
                success,image = vidcap.read()

                if not success: continue

                #print(pathWrite)
                cv2.imwrite(os.path.join(pathWrite, "frame%d.jpg" % count), image)
                count = count + 1
            idx = idx + 1


def findHyperKvasirMask(height, width, toEdgeMask=False,itr=1):
    sampled_path1 = os.path.join('masks','sampled_mask_hpk4.jpg')

    original = cv2.imread(sampled_path1)
    original = cv2.resize(original, (height,width))
    img = cv2.blur(original, (10,10))
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th, im_gray_th_otsu = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), 'uint8')
    img = cv2.erode(im_gray_th_otsu, kernel, iterations=itr)
    img = cv2.subtract(255,img)
    mask = np.stack((img,)*3, axis=-1)

    if toEdgeMask:
        edges = cv2.Canny(img, threshold1=100,threshold2=200)
        edges = cv2.dilate(edges, kernel, iterations=1)
        mask = np.stack((edges,)*3, axis=-1)
    
    return mask


def findSunMask(height, width, toEdgeMask=False,itr=1):
    sampled_path2 = os.path.join('masks','sampled_mask_sun.jpg')

    original = cv2.imread(sampled_path2)
    original = cv2.resize(original, (height,width))
    img = cv2.blur(original, (10,10))
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th, im_gray_th_otsu = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), 'uint8')
    img = cv2.erode(im_gray_th_otsu, kernel, iterations=itr)
    img = cv2.subtract(255,img)
    mask = np.stack((img,)*3, axis=-1)

    if toEdgeMask:
        edges = cv2.Canny(img, threshold1=100,threshold2=200)
        edges = cv2.dilate(edges, kernel, iterations=1)
        mask = np.stack((edges,)*3, axis=-1)

    return mask

