from torch.utils.data import DataLoader
from dataloaders.datasets import train_data, val_data
import pdb
import torch

def make_data_loader(args, **kwargs):
        ############################ kitti15 ###########################
        # if args.dataset == 'kitti15':              
        #     train_list= 'dataloaders/lists/kitti2015_train180.list'
        #     test_list = 'dataloaders/lists/kitti2015_val20.list'  
        #     train_set = stereo.DatasetFromList(args, train_list, [args.crop_height, args.crop_width], True)
        #     test_set  = stereo.DatasetFromList(args, test_list,  [384,1248], False)
           
        #     train_loader= DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        #     test_loader = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)
        #     return train_loader, test_loader

        ############################ kitti12 ###########################
        if args.dataset == 'kitti12':              
            # train_list= 'dataloaders/lists/kitti2012_train170.list'
            # test_list = 'dataloaders/lists/kitti2012_val24.list'  
            # train_set = stereo.DatasetFromList(args, train_list, [args.crop_height, args.crop_width], True)
            # test_set  = stereo.DatasetFromList(args, test_list,  [384,1248], False)
            if args.stage == 'search':
                crop_size = [240, 240]
                k12_train_data_dir = '/home/featurize/data/kitti2012_train'
                k15_train_data_dir = '/home/featurize/data/kitti15_train'
                single_stereo = True
                
                train_set = train_data.K12_dataset(crop_size=crop_size, K12root=k12_train_data_dir, 
                                                    K15root=k15_train_data_dir, single_stereo=single_stereo)

                # print(len(train_set))
                train_setA = torch.utils.data.Subset(train_set, range(100))
                train_setB = torch.utils.data.Subset(train_set, range(100, 200))
                # train_setA = torch.utils.data.Subset(train_set, range(10))
                # train_setB = torch.utils.data.Subset(train_set, range(10, 20))
                # print(len(train_setB))

                train_loaderA= DataLoader(train_setA, batch_size=args.batch_size, shuffle=True, **kwargs)
                train_loaderB= DataLoader(train_setB, batch_size=args.batch_size, shuffle=True, **kwargs)

                k12_test_data_dir = '/home/featurize/data/kitti12_testing'
                k15_test_data_dir = '/home/featurize/data/kitti15_test'

                test_set = val_data.K12_testloader(K12root=k12_test_data_dir, K15root=k15_test_data_dir, 
                                                    crop_size=crop_size, single_stereo=single_stereo)

                test_loader = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)

                # val_set = torch.utils.data.Subset(test_set, range(1000))
                val_set = torch.utils.data.Subset(test_set, range(100))
                val_loader = DataLoader(val_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)

                return train_loaderA, train_loaderB, val_loader, test_loader 
            elif args.stage == 'train':
                crop_size = [240, 240]
                # k12_train_data_dir = '/home/featurize/data/kitti2012_train'
                k15_train_data_dir = '/home/featurize/data/kitti15_train'
                single_stereo = True
                
                train_set = train_data.K12_dataset(crop_size=crop_size, K12root=None, 
                                                    K15root=k15_train_data_dir, single_stereo=single_stereo)
            
                train_loader= DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

                k12_test_data_dir = '/home/featurize/data/kitti12_testing'
                k15_test_data_dir = '/home/featurize/data/kitti15_test'

                test_set = val_data.K12_testloader(K12root=None, K15root=k15_test_data_dir, crop_size=crop_size,
                                                         single_stereo=single_stereo)

                test_loader = DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)
                
                val_set = torch.utils.data.Subset(test_set, range(1000))
                val_loader = DataLoader(val_set, batch_size=args.testBatchSize, shuffle=False, **kwargs)
                return train_loader, test_loader

        else:
            raise NotImplementedError
