# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop
import os
import cv2
from torch.utils.tensorboard import SummaryWriter


# --- Training dataset --- #

def get_dataset(dir):
    if not os.path.isdir(dir):
        raise Exception('check' + dir)
    image_list = sorted(os.listdir(dir))
    images = []
    for img in image_list:
        images.append(dir + '/' + img)
    return images
    
class K12_dataset(data.Dataset):
    """Some Information about K12(K15)_dataset"""

    def __init__(self, K12root, K15root, crop_size, single_stereo=True):
        super(K12_dataset, self).__init__()
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

        self.crop_size = crop_size
        self.K12root = K12root
        self.single_stereo = single_stereo



    def __getitem__(self, index):
        crop_width, crop_height = self.crop_size
        gt_img = Image.open(self.gt_images[index]).convert('RGB')
        haze_img = Image.open(self.rain_images[index]).convert('RGB')
        
        tmp = index     
        if index < self.len_k12:
            tmp = 2 * index
        else:
            tmp = index + self.len_k12

        if self.single_stereo:
            gt_img2 = Image.open(self.gt_images2[index]).convert('RGB')
            haze_img2 = Image.open(self.rain_images2[tmp]).convert('RGB')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        if self.single_stereo:
            haze_crop_img2 = haze_img2.crop((x, y, x + crop_width, y + crop_height))
            gt_crop_img2 = gt_img2.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)
        if self.single_stereo:
            haze2 = transform_haze(haze_crop_img2)
            gt2 = transform_gt(gt_crop_img2)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        if self.single_stereo:
            return haze, haze2, gt, gt2
            
        return haze, gt


    def __len__(self):
        return len(self.gt_images)


# class rain_cityscape_dataset(data.Dataset):
#     def __init__(self, img_root, gt_root,  crop_size):
#         super(rain_cityscape_dataset, self).__init__()
#         self.img_root = img_root
#         self.gt_root = gt_root
#         self.crop_size = crop_size
#         self.rain_images, self.gt_images = self.make_city_dataset(img_root, gt_root)


#     def make_city_dataset(self, img_root, gt_root):
#         names = os.listdir(img_root)
#         print("get cities : ", names)
#         img_list = []
#         gt_list = []
#         for name in names:
#             imgs = os.listdir(img_root + '/' + name)
#             for img in imgs:
#                 img_list.append(name + '/' + img)
#                 gt = img.split("_")
#                 # print(gt)
#                 gt = '_'.join(gt[:4]) + '.png'
#                 gt_list.append(name + '/' + gt)
#         print("---- img number :{} ----".format(len(img_list)))
#         return img_list, gt_list 

#     def __getitem__(self, index):
#         crop_width, crop_height = self.crop_size
#         gt_img = Image.open(self.gt_root + '/' + self.gt_images[index]).convert('RGB')
#         haze_img = Image.open(self.img_root + '/' + self.rain_images[index]).convert('RGB')

#         width, height = haze_img.size

#         if width < crop_width or height < crop_height:
#             raise Exception('Bad image size: {}'.format(gt_name))

#         # --- x,y coordinate of left-top corner --- #
#         x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
#         haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
#         gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

#         # --- Transform to tensor --- #
#         transform_haze = Compose([ToTensor()])
#         transform_gt = Compose([ToTensor()])
#         haze = transform_haze(haze_crop_img)
#         gt = transform_gt(gt_crop_img)

#         # --- Check the channel is 3 or not --- #
#         if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
#             raise Exception('Bad image channel: {}'.format(gt_name))

            
#         return haze, gt


#     def __len__(self):
#         return len(self.gt_images)

if __name__ == "__main__":
    

    crop_size = [240, 240]
    k12_train_data_dir = '/home/featurize/data/kitti2012_train'
    k15_train_data_dir = '/home/featurize/data/kitti15_train'
    single_stereo = True
    data_set = K12_dataset(crop_size=crop_size, K12root=k12_train_data_dir, 
                                                    K15root=k15_train_data_dir, single_stereo=single_stereo)


    # print(data_set.gt_images[5500])
    # print(data_set.gt_images2[5500])
    # print(data_set.rain_images[5500])
    # print(data_set.rain_images2[5500])
    # gt_img = Image.open(data_set.gt_images[2500]).convert('RGB')
    # haze_img = Image.open(data_set.rain_images[2500]).convert('RGB')
    # gt2_img = Image.open(data_set.gt_images2[2500]).convert('RGB')
    # haze2_img = Image.open(data_set.rain_images2[2500]).convert('RGB')

    
    # print(data_set.gt_images[5])
    # print(data_set.gt_images2[5])
    # print(data_set.rain_images[5])
    # print(data_set.rain_images2[10])
    # gt_img = Image.open(data_set.gt_images[5]).convert('RGB')
    # haze_img = Image.open(data_set.rain_images[5]).convert('RGB')
    # gt2_img = Image.open(data_set.gt_images2[5]).convert('RGB')
    # haze2_img = Image.open(data_set.rain_images2[10]).convert('RGB')

    # gt_img.save('gt_img.png')
    # haze_img.save('haze_img.png')
    # gt2_img.save('gt2_img.png')
    # haze2_img.save('haze2_img.png')

    writer = SummaryWriter("dataloader")    
    train_data_loader = data.DataLoader(data_set,
                                        batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    for batch_id, train_data in enumerate(train_data_loader):
        if single_stereo:
            haze, haze2, gt, gt2 = train_data

            writer.add_images("haze", haze, batch_id)
            writer.add_images("haze2", haze2, batch_id)
            writer.add_images("gt", gt, batch_id)
            writer.add_images("gt2", gt2, batch_id)

        else:
            haze, gt = train_data
            print(haze.shape)
    writer.close()
