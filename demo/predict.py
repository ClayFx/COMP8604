import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from retrain.Restormer import StereoRestormer
from torchvision.transforms import Compose, ToTensor, CenterCrop, ToPILImage

cuda = True

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without cuda")

transform_haze = Compose([ToTensor(), CenterCrop((240, 240))])

transform_target = Compose([ToPILImage()])

model = StereoRestormer(inp_channels=6, 
        out_channels=6, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False)       ## True for dual-pixel defocus deblurring only. Also set inp_channels=6

model = torch.nn.DataParallel(model).cuda()

checkpoint = torch.load("retrain/pretrain/best_epoch_19-34.70.pth")
model.load_state_dict(checkpoint['state_dict'])

left_img = Image.open('demo/test_left.jpg')
right_img = Image.open("demo/test_right.jpg")


if __name__ == "__main__":
    model.eval()
    left_img = transform_haze(left_img)
    right_img = transform_haze(right_img)
    stereo_img = torch.cat((left_img, right_img), dim=0).unsqueeze(0).cuda()
    output = model(stereo_img)
    
    output_1, output_2 = output[0,0:3,:,:], output[0,3:,:,:]
    output_1 = transform_target(output_1)
    output_2 = transform_target(output_2)
    output_1.save("demo/output_left.jpg")
    output_2.save("demo/output_right.jpg")
