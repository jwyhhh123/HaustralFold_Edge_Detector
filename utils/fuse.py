import cv2
import torch
import numpy as np

def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img

def fuse(tensor, args, mask = None):

        # 255.0 * (1.0 - em_a)
        edge_maps = []
        for i in tensor:
            tmp = torch.sigmoid(i).cpu().detach().numpy()
            edge_maps.append(tmp)
        tensor = np.array(edge_maps)
        # print(f"tensor shape: {tensor.shape}")

        idx = 0
        fused_tensor = []
        for k in range(tensor.shape[1]):
            tmp = tensor[:, idx, ...]
            # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
            tmp = np.squeeze(tmp)

            # Iterate our all 7 NN outputs for a particular image
            for i in range(tmp.shape[0]):
                tmp_img = tmp[i]
                tmp_img = np.uint8(image_normalization(tmp_img))
                tmp_img = cv2.bitwise_not(tmp_img)

                # Resize prediction to match input image size
                if not tmp_img.shape[1] == args.img_width or not tmp_img.shape[0] == args.img_height:
                    tmp_img = cv2.resize(tmp_img, (args.img_width, args.img_height))

                if i == 6:
                    fuse = tmp_img
                    fuse = fuse.astype(np.uint8)

            #fuse = cv2.add(fuse, mask)
            fuse = torch.unsqueeze(torch.from_numpy(fuse),0)
            fused_tensor.append(fuse)

            idx += 1

        if len(fused_tensor) == 1: return fused_tensor[0]
        
        return torch.cat(fused_tensor)
