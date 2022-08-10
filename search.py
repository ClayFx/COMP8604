import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
from mypath import Path
from dataloaders import make_data_loader
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.copy_state_dict import copy_state_dict
from torch.autograd import Variable
from time import time
import imageio
import apex
import torch.nn.functional as F
import pdb
from config_utils.search_args import obtain_search_args
from models.build_model import AutoStereo
from utils.utils import stereo_validation, to_psnr, to_ssim_skimage

print('working with pytorch version {}'.format(torch.__version__))
print('with cuda version {}'.format(torch.version.cuda))
print('cudnn enabled: {}'.format(torch.backends.cudnn.enabled))
print('cudnn version: {}'.format(torch.backends.cudnn.version()))


opt = obtain_search_args()
print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

# default settings for epochs, batch_size and lr
if opt.epochs is None:
    epoches = {'sceneflow': 10, 'middlebury' : 10, 'kitti12' : 2}
    opt.epochs = epoches[opt.dataset.lower()]

if opt.batch_size is None:
        opt.batch_size = 4 * len(opt.gpu_ids)

if opt.testBatchSize is None:
        opt.testBatchSize = opt.batch_size


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last':True}

        self.train_loaderA, self.train_loaderB, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)

        # Define network
        model = AutoStereo(maxdisp = self.args.max_disp, 
                           Fea_Layers=self.args.fea_num_layers, Fea_Filter=self.args.fea_filter_multiplier, 
                           Fea_Block=self.args.fea_block_multiplier, Fea_Step=self.args.fea_step, 
                           Mat_Layers=self.args.mat_num_layers, Mat_Filter=self.args.mat_filter_multiplier, 
                           Mat_Block=self.args.mat_block_multiplier, Mat_Step=self.args.mat_step)

        optimizer_F = torch.optim.SGD(
                model.feature.weight_parameters(), 
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )        
        optimizer_M = torch.optim.SGD(
                model.matching.weight_parameters(), 
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )

 
        self.model, self.optimizer_F, self.optimizer_M = model, optimizer_F, optimizer_M       
        self.architect_optimizer_F = torch.optim.Adam(self.model.feature.arch_parameters(),
                                                    lr=args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=args.arch_weight_decay)

        self.architect_optimizer_M = torch.optim.Adam(self.model.matching.arch_parameters(),
                                                    lr=args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=args.arch_weight_decay)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loaderA), min_lr=args.min_lr)
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()

        # Resuming checkpoint
        self.best_pred = float("-inf")
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            # if the weights are wrapped in module object we have to clean it
            if args.clean_module:
                self.model.load_state_dict(checkpoint['state_dict'])
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.find('module') != -1:
                        print(1)
                        pdb.set_trace()
                        name = k[7:]  # remove 'module.' of dataparallel
                        new_state_dict[name] = v
                # self.model.load_state_dict(new_state_dict)
                pdb.set_trace()
                copy_state_dict(self.model.state_dict(), new_state_dict)

            else:
                if torch.cuda.device_count() > 1:#or args.load_parallel:
                    # self.model.module.load_state_dict(checkpoint['state_dict'])
                    copy_state_dict(self.model.module.state_dict(), checkpoint['state_dict'])
                else:
                    # self.model.load_state_dict(checkpoint['state_dict'])
                    copy_state_dict(self.model.module.state_dict(), checkpoint['state_dict'])


            if not args.ft:
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                copy_state_dict(self.optimizer_M.state_dict(), checkpoint['optimizer_M'])
                copy_state_dict(self.optimizer_F.state_dict(), checkpoint['optimizer_F'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        print('Total number of model parameters : {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))
        print('Number of Feature Net parameters: {}'.format(sum([p.data.nelement() for p in self.model.module.feature.parameters()])))
        print('Number of Matching Net parameters: {}'.format(sum([p.data.nelement() for p in self.model.module.matching.parameters()])))


    def training(self, epoch):
        train_loss = 0.0
        valid_iteration = 0
        self.model.train()
        tbar = tqdm(self.train_loaderA)
        num_img_tr = len(self.train_loaderA)

        for i, batch in enumerate(tbar):
            input1, input2, target1, target2 = Variable(batch[0],requires_grad=True), Variable(batch[1], requires_grad=True), \
                                                 (batch[2]), (batch[3]) 
            if cuda:
                input1 = input1.cuda()
                input2 = input2.cuda()
                target1 = target1.cuda()
                target2 = target2.cuda()
            
            target = torch.cat((target1, target2), dim=1)
            target = torch.squeeze(target, 1)

            # mask = target < self.args.max_disp
            # mask.detach_()
            # valid = target[mask].size()[0]
            if True:
                self.scheduler(self.optimizer_F, i, epoch, self.best_pred)
                self.scheduler(self.optimizer_M, i, epoch, self.best_pred)
                self.optimizer_F.zero_grad()
                self.optimizer_M.zero_grad()
            
                output = self.model(input1, input2) 
                output1, output2 = output[:,0:3,:,:], output[:,3:,:,:]
                # print(output.size(), target.size())
                # TODO -- change l1 loss to l2 loss
                loss = F.mse_loss(output1, target1, reduction='mean') + F.mse_loss(output2, target2, reduction='mean')
                loss.backward()            
                self.optimizer_F.step()     
                self.optimizer_M.step()   

                if epoch >= self.args.alpha_epoch:
                    print("Start searching architecture!...........")
                    search = next(iter(self.train_loaderB))
                    input1_search, input2_search, target1_search, target2_search = Variable(search[0],requires_grad=True), \
                                                                Variable(search[1], requires_grad=True), (search[2]), (search[3])
                    if self.args.cuda:
                        input1_search = input1_search.cuda()
                        input2_search = input2_search.cuda()
                        target1_search = target1_search.cuda()
                        target2_search = target2_search.cuda()

                    target_search=torch.cat((target1_search, target2_search), dim=1)
                    target_search=torch.squeeze(target_search,1)

                    # mask_search = target_search < self.args.max_disp
                    # mask_search.detach_()

                    self.architect_optimizer_F.zero_grad()
                    self.architect_optimizer_M.zero_grad()
                    output_search = self.model(input1_search, input2_search)
                    output_search1, output_search2 = output_search[:,0:3,:,:], output_search[:,3:,:,:]

                    arch_loss = F.mse_loss(output_search1, target1_search, reduction='mean') + \
                            F.mse_loss(output_search2, target2_search, reduction='mean')

                    arch_loss.backward()            
                    self.architect_optimizer_F.step() 
                    self.architect_optimizer_M.step()   

                train_loss += loss.item()
                valid_iteration += 1
                tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            #Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image_stereo(self.writer, input1, input2, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print("=== Train ===> Epoch :{} Error: {:.4f}".format(epoch, train_loss/valid_iteration))
        print(self.model.module.feature.alphas)

        #save checkpoint every epoch
        is_best = False
        if torch.cuda.device_count() > 1:
           state_dict = self.model.module.state_dict()
        else:
           state_dict = self.model.state_dict()

        self.saver.save_checkpoint({
               'epoch': epoch + 1,
               'state_dict': state_dict,
               'optimizer_F': self.optimizer_F.state_dict(),
               'optimizer_M': self.optimizer_M.state_dict(),
               'best_pred': self.best_pred,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch))

    def validation(self, epoch):
        self.model.eval()
        
        epoch_error = 0
        three_px_acc_all = 0
        valid_iteration = 0

        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        psnr_list = []
        ssim_list = []
        psnr_list2 = []
        ssim_list2 = []

        for i, batch in enumerate(tbar):
            input1, input2, target1, target2 = Variable(batch[0],requires_grad=False), Variable(batch[1], requires_grad=False),\
                                     Variable(batch[2], requires_grad=False), Variable(batch[3], requires_grad=False)
            if self.args.cuda:
                input1 = input1.cuda()
                input2 = input2.cuda()
                target1 = target1.cuda()
                target2 = target2.cuda()
                
            
            target = torch.cat((target1, target2), dim=1)
            target = torch.squeeze(target, 1)

            # mask = target < self.args.max_disp
            # mask.detach_()
            # valid = target[mask].size()[0]

            # if valid>0:
            if True:
                with torch.no_grad():
                    output = self.model(input1, input2)

                    output1, output2 = output[:,0:3,:,:], output[:,3:,:,:]
                    error = torch.mean(torch.abs(output1 - target1)) + torch.mean(torch.abs(output2 - target2))
                    epoch_error += error.item()

                    valid_iteration += 1

                    #computing 3-px error#                
                    # pred_disp = output.cpu().detach()                                                                                                                          
                    # true_disp = target.cpu().detach()
                    # disp_true = true_disp
                    # index = np.argwhere(true_disp<opt.max_disp)
                    # disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
                    # correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 1)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
                    # three_px_acc = 1-(float(torch.sum(correct))/float(len(index[0])))

                    # three_px_acc_all += three_px_acc

                    this_psnr1 = to_psnr(output1, target1)
                    this_ssim1 = to_ssim_skimage(output1, target1)
                    this_psnr2 = to_psnr(output2, target2)
                    this_ssim2 = to_ssim_skimage(output2, target2)

                    psnr_list.append(this_psnr1)
                    ssim_list.append(this_ssim1)
                    psnr_list2.append(this_psnr2)
                    ssim_list2.append(this_ssim2)

                    this_psnr = (np.mean(this_psnr1) + np.mean(this_psnr2)) / 2 
                    this_ssim = (np.mean(this_ssim1) + np.mean(this_ssim2)) / 2 

                    print("===> Test({}/{}): Error(EPE, psnr, ssim): ({:.4f} {:.4f} {:.4f})".format(i, len(self.val_loader), error.item(), this_psnr, this_ssim))

        self.writer.add_scalar('val/EPE', epoch_error/valid_iteration, epoch)
        self.writer.add_scalar('val/psnr', three_px_acc_all/valid_iteration, epoch)
        self.writer.add_scalar('val/ssim', three_px_acc_all/valid_iteration, epoch)


        avg_psnr = (np.mean(psnr_list) + np.mean(psnr_list2)) / 2
        avg_ssim = (np.mean(ssim_list) + np.mean(ssim_list2)) / 2
        print("===> Test: Avg. Error(EPE, psnr, ssim): ({:.4f} {:.4f} {:.4f})".format(epoch_error/valid_iteration, avg_psnr, avg_ssim))


        # save model
        new_pred = avg_psnr
        if new_pred > self.best_pred: 
            is_best = True
            self.best_pred = new_pred
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer_F': self.optimizer_F.state_dict(),
                'optimizer_M': self.optimizer_M.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

if __name__ == "__main__":
   
    trainer = Trainer(opt)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val:
            trainer.validation(epoch)

    trainer.writer.close()
