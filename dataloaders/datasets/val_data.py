# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import os
from random import randrange

# --- Validation/test dataset --- #

def get_dataset(dir):
    if not os.path.isdir(dir):
        raise Exception('check' + dir)
    image_list = sorted(os.listdir(dir))
    images = []
    for img in image_list:
        images.append(dir + '/' + img)
    return images

class K12_testloader(data.Dataset):
    """Some Information about K12_testloader"""
    # def __init__(self, root, single_stereo=True):
    #     super(K12_testloader, self).__init__()
    #     self.gt_images = get_dataset(root + '/image_2_3_norain')
    #     self.rain_images = get_dataset(root + '/image_2_3_rain50')
    #     self.gt_images2 = get_dataset(root + '/image_3_2_norain')
    #     self.rain_images2 = get_dataset(root + '/image_3_2_rain50')
    #     self.root = root
    #     self.single_stereo = single_stereo
    #     # print(self.gt_images)

    def __init__(self, K12root, K15root, crop_size, single_stereo=True):
        super(K12_testloader, self).__init__()
        # ---- root == './' ---
        if K12root is not None:
            self.gt_images = get_dataset(K12root + '/image_2')
            self.rain_images = get_dataset(K12root + '/image_2_rain50')
            self.gt_images2 = get_dataset(K12root + '/image_3')
            self.rain_images2 = get_dataset(K12root + '/image_3_2_rain50')
            self.len_k12 = len(self.gt_images)
        else:
            self.gt_images = []
            self.rain_images = []
            self.gt_images2 = []
            self.rain_images2 = []
            self.len_k12 = 0

        #K15 Images
        self.gt_images.extend(get_dataset(K15root + '/image_2'))
        self.rain_images.extend(get_dataset(K15root + '/image_2_rain50'))
        self.gt_images2.extend(get_dataset(K15root + '/image_3'))
        self.rain_images2.extend(get_dataset(K15root + '/image_3_rain50'))

        self.K12root = K12root
        self.single_stereo = single_stereo
        self.crop_size = crop_size

    def __getitem__(self, index):
        crop_width, crop_height = self.crop_size
        gt = Image.open(self.gt_images[index])
        haze = Image.open(self.rain_images[index])

        tmp = index           
        if index < self.len_k12:
            tmp = 2 * index
        else:
            tmp = index + self.len_k12    
                                      
        if self.single_stereo:
            gt2 = Image.open(self.gt_images2[index])
            haze2 = Image.open(self.rain_images2[tmp])


        gt = gt.convert('RGB')
        haze = haze.convert('RGB')
        if self.single_stereo:
            gt2 = gt2.convert('RGB')
            haze2 = haze2.convert('RGB')

        width, height = haze.size

        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt.crop((x, y, x + crop_width, y + crop_height))
        if self.single_stereo:
            haze_crop_img2 = haze2.crop((x, y, x + crop_width, y + crop_height))
            gt_crop_img2 = gt2.crop((x, y, x + crop_width, y + crop_height))

        transform_haze = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])

        gt = transform_gt(gt_crop_img)
        haze = transform_haze(haze_crop_img)
        if self.single_stereo:
            gt2 = transform_gt(gt_crop_img2)
            haze2 = transform_haze(haze_crop_img2)

        if self.single_stereo:
            return haze, haze2, gt, gt2, self.rain_images[index], self.rain_images2[tmp]
        else:
            return haze, gt, self.rain_images[index]
            
        # if self.single_stereo:
        #     return haze, haze2, gt, gt2
        # else:
        #     return haze, gt

    def __len__(self):
        return len(self.gt_images)


# class rain_cityscape_dataset(data.Dataset):
#     def __init__(self, img_root, gt_root):
#         super(rain_cityscape_dataset, self).__init__()
#         self.img_root = img_root
#         self.gt_root = gt_root
#         self.rain_images, self.gt_images = self.__make_city_dataset(img_root)


#     def __make_city_dataset(self, img_root):
#         names = os.listdir(img_root)
#         print("get cities : ", names)
#         img_list = []
#         gt_list = []
#         for name in names:
#             imgs = os.listdir(img_root + '/' + name)
#             for img in imgs:
#                 img_list.append(name + '/' + img)
#                 gt = img.split("_")
#                 gt = '_'.join(gt[:4]) + '.png'
#                 gt_list.append(name + '/' + gt)
#         print("---- img number :{} ----".format(len(img_list)))
#         return img_list, gt_list 

#     def __getitem__(self, index):

#         gt_img = Image.open(self.gt_root + '/' + self.gt_images[index]).convert('RGB')
#         haze_img = Image.open(self.img_root + '/' + self.rain_images[index]).convert('RGB')


#         # --- Transform to tensor --- #
#         transform_haze = Compose([ToTensor()])
#         transform_gt = Compose([ToTensor()])
#         haze = transform_haze(haze_img)
#         gt = transform_gt(gt_img)

#         # --- Check the channel is 3 or not --- #
#         if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
#             raise Exception('Bad image channel: {}'.format(gt_name))

            
#         return haze, gt, self.rain_images[index]

#     def __len__(self):
#         return len(self.gt_images)