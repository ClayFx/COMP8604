CUDA_VISIBLE_DEVICES=0,1 
python train.py \
                --batch_size=4 \
                --testBatchSize=16 \
                --crop_height=240 \
                --crop_width=240 \
                --maxdisp=256 \
                --threads=6 \
                --dataset='kitti12' \
                --save_path='./run/deraining/test/' \
                --nEpochs=800 2>&1 |tee ./run/deraining/test/log.txt
                # --save_path='./run/deraining/Kitti12_train/' \
                # --save_path='./run/deraining/use_conv2d/' \
                # --resume='/home/featurize/work/LEAStereo/run/deraining/Restormer/best_epoch_4.pth' \
                # --resume='/home/featurize/work/LEAStereo/run/deraining/Kitti12_train/best_epoch_3.pth' \


