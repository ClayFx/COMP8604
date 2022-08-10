from __future__ import print_function
import argparse
from math import log10

import sys
import shutil
import os
import dataloaders
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import skimage
import pdb
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from time import time
from collections import OrderedDict
# from retrain.LEAStereo import LEAStereo
from retrain.Restormer import StereoRestormer

from mypath import Path
from dataloaders import make_data_loader
from utils.multadds_count import count_parameters_in_MB, comp_multadds, comp_multadds_fw
from config_utils.train_args import obtain_train_args
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.utils import stereo_validation


opt = obtain_train_args()
print(opt)

cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
kwargs = {'num_workers': opt.threads, 'pin_memory': True, 'drop_last':True}
training_data_loader, testing_data_loader = make_data_loader(opt, **kwargs)

print('===> Building model')
# model = LEAStereo(opt)
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

writer = SummaryWriter(opt.save_path) 

## compute parameters
#print('Total number of model parameters : {}'.format(sum([p.data.nelement() for p in model.parameters()])))
#print('Number of Feature Net parameters: {}'.format(sum([p.data.nelement() for p in model.feature.parameters()])))
#print('Number of Matching Net parameters: {}'.format(sum([p.data.nelement() for p in model.matching.parameters()])))

print('Total Params = %.2fMB' % count_parameters_in_MB(model))
# print('Feature Net Params = %.2fMB' % count_parameters_in_MB(model.feature))
# print('Matching Net Params = %.2fMB' % count_parameters_in_MB(model.matching))
   
#mult_adds = comp_multadds(model, input_size=(3,opt.crop_height, opt.crop_width)) #(3,192, 192))
#print("compute_average_flops_cost = %.2fMB" % mult_adds)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

torch.backends.cudnn.benchmark = True

if opt.solver == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9,0.999))
elif opt.solver == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

def train(epoch):
    epoch_loss = 0
    epoch_error = 0
    valid_iteration = 0
    tbar = tqdm(training_data_loader)
    
    for iteration, batch in enumerate(tbar):
        input1, input2, target1, target2 = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), \
                                        (batch[2]), (batch[3])
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target1 = target1.cuda()
            target2 = target2.cuda()

        target = torch.cat((target1, target2), dim=1)
        target = torch.squeeze(target, 1)

        input_img = torch.cat((input1, input2), dim=1)
        # print(input_img.size())
        # target2=torch.squeeze(target2,1)
        # mask = target < opt.maxdisp
        # mask.detach_()
        # valid = target[mask].size()[0]
        train_start_time = time()
        # if valid > 0:
        if True:
            model.train()
    
            optimizer.zero_grad()
            # disp = model(input1,input2) 
            disp = model(input_img) 
            # TODO -- change l1 loss to l2 loss
            # loss = F.smooth_l1_loss(disp[mask], target[mask], reduction='mean')

            output_1, output_2 = disp[:,0:3,:,:], disp[:,3:,:,:]
            loss = F.l1_loss(output_1, target1, reduction='mean') + F.l1_loss(output_2, target2, reduction='mean')
            # loss = F.mse_loss(output_1, target1, reduction='mean') +  F.mse_loss(output_2, target2, reduction='mean') 
            loss.backward()
            optimizer.step()
            
            error = torch.mean(torch.abs(disp - target)) 
            train_end_time = time()
            train_time = train_end_time - train_start_time
            
            if iteration % (len(training_data_loader) // 10) == 0:
                global_step = iteration + len(training_data_loader) * epoch
                writer.add_images("input1", input1[0], global_step, dataformats='CHW')
                writer.add_images("input2", input2[0], global_step, dataformats='CHW')
                writer.add_images("target1", target1[0], global_step, dataformats='CHW')
                writer.add_images("target2", target2[0], global_step, dataformats='CHW')
                writer.add_images("output_1", output_1[0], global_step, dataformats='CHW')
                writer.add_images("output_2", output_2[0], global_step, dataformats='CHW')

            epoch_loss += loss.item()
            valid_iteration += 1
            epoch_error += error.item()
            tbar.set_description("===> Epoch[{}]({}/{}): Loss: ({:.4f}), Error: ({:.4f}), Time: ({:.2f}s)".format(epoch, iteration, len(training_data_loader), loss.item(), error.item(), train_time))
            # print("===> Epoch[{}]({}/{}): Loss: ({:.4f}), Error: ({:.4f}), Time: ({:.2f}s)".format(epoch, iteration, len(training_data_loader), loss.item(), error.item(), train_time))
            sys.stdout.flush()        
            writer.add_scalar('train/total_loss_iteration', loss.item(), iteration + len(training_data_loader) * epoch) 
                          
    print("===> Epoch {} Complete: Avg. Loss: ({:.4f}), Avg. Error: ({:.4f})".format(epoch, epoch_loss / valid_iteration, epoch_error/valid_iteration))
    writer.add_scalar('train/total_loss_epoch', epoch_loss / valid_iteration, epoch) 

def val():
    epoch_error = 0
    valid_iteration = 0
    three_px_acc_all = 0
    model.eval()
    for iteration, batch in enumerate(testing_data_loader):
        input1, input2, target1, target2 = Variable(batch[0],requires_grad=False), Variable(batch[1], requires_grad=False), \
                                         Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target1 = target1.cuda()
            target2 = target2.cuda()

        target = torch.cat((target1, target2), dim=1)
        target = torch.squeeze(target, 1)

        # mask = target < opt.maxdisp
        # mask.detach_()
        # valid=target[mask].size()[0]
        if True:
            with torch.no_grad(): 
                disp = model(input1,input2)
                error = torch.mean(torch.abs(disp - target)) 

                valid_iteration += 1
                epoch_error += error.item()              

                #computing 3-px error#                
                # pred_disp = disp.cpu().detach() 
                # true_disp = target.cpu().detach()
                # disp_true = true_disp
                # index = np.argwhere(true_disp<opt.maxdisp)
                # disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
                # correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 1)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
                # three_px_acc = 1-(float(torch.sum(correct))/float(len(index[0])))

                # three_px_acc_all += three_px_acc
    
                # print("===> Test({}/{}): Error: ({:.4f} {:.4f})".format(iteration, len(testing_data_loader), error.item(), three_px_acc))
                print("===> Test({}/{}): Error: ({:.4f})".format(iteration, len(testing_data_loader), error.item()))
                sys.stdout.flush()

    # print("===> Test: Avg. Error: ({:.4f} {:.4f})".format(epoch_error/valid_iteration, three_px_acc_all/valid_iteration))
    # return three_px_acc_all/valid_iteration
    return epoch_error/valid_iteration

def save_checkpoint(save_path, epoch, state, is_best):
    filename = save_path + "epoch_{}.pth".format(epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + f'best_epoch_{epoch}.pth')
    print("Checkpoint saved to {}".format(filename))

if __name__ == '__main__':
    if opt.resume is not None:
        # if not os.path.isfile(opt.resume):
        #     raise RuntimeError("=> no checkpoint found at '{}'" .format(opt.resume))
        # checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        opt.start_epoch = 0

    error=float("-inf")
    for epoch in range(opt.start_epoch, opt.nEpochs):
        # train(epoch)
        is_best = False
        # loss=val()
        psnr1, ssim1, psnr2, ssim2 = stereo_validation(model, testing_data_loader) 
        loss = (psnr1+psnr2)/2
        print(f"Epoch {epoch} : AVG. PSNR {loss}")
        if loss > error:
            error=loss
            is_best = True
        if opt.dataset == 'sceneflow':
            if epoch>=0:
                save_checkpoint(opt.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)
        else:
            # if epoch%100 == 0 and epoch >= 3000:
            if epoch>=0:
                save_checkpoint(opt.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)
            if is_best:
                save_checkpoint(opt.save_path, epoch,{
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best)

        scheduler.step()

    writer.close()

    save_checkpoint(opt.save_path, opt.nEpochs,{
            'epoch': opt.nEpochs,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)
