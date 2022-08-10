CUDA_VISIBLE_DEVICES=0,1 
python train.py \
                --batch_size=8 \
                --testBatchSize=16 \
                --crop_height=240 \
                --crop_width=240 \
                --maxdisp=256 \
                --threads=6 \
                --dataset='kitti12' \
                --save_path='./run/deraining/Restormer_test/' \
                --fea_num_layer 6 --mat_num_layers 12 \
                --fea_filter_multiplier 8 --fea_block_multiplier 4 --fea_step 3  \
                --mat_filter_multiplier 8 --mat_block_multiplier 4 --mat_step 3  \
                --net_arch_fea='run/fromLeastereo/sceneflow/best/architecture/feature_network_path.npy' \
                --cell_arch_fea='run/fromLeastereo/sceneflow/best/architecture/feature_genotype.npy' \
                --net_arch_mat='run/fromLeastereo/sceneflow/best/architecture/matching_network_path.npy' \
                --cell_arch_mat='run/fromLeastereo/sceneflow/best/architecture/matching_genotype.npy' \
                --resume='/home/featurize/work/LEAStereo/run/deraining/Restormer_bs8/best_epoch_19.pth' \
                --nEpochs=800 2>&1 |tee ./run/deraining/Restormer_test/log.txt
                
                # --save_path='./run/deraining/Kitti12_train/' \
                # --save_path='./run/deraining/use_conv2d/' \
                # --resume='/home/featurize/work/LEAStereo/run/deraining/Restormer/best_epoch_4.pth' \
                # --resume='/home/featurize/work/LEAStereo/run/deraining/Kitti12_train/best_epoch_3.pth' \

               #--resume='./run/Kitti12/best/best_1.16.pth'

                # --net_arch_fea='/home/featurize/work/LEAStereo/run/deraining/kitti12/feature_network_path.npy' \
                # --cell_arch_fea='/home/featurize/work/LEAStereo/run/deraining/kitti12/feature_genotype.npy' \
                # --net_arch_mat='/home/featurize/work/LEAStereo/run/deraining/kitti12/matching_network_path.npy' \
                # --cell_arch_mat='/home/featurize/work/LEAStereo/run/deraining/kitti12/matching_genotype.npy' \

                # --net_arch_fea='run/fromLeastereo/sceneflow/best/architecture/feature_network_path.npy' \
                # --cell_arch_fea='run/fromLeastereo/sceneflow/best/architecture/feature_genotype.npy' \
                # --net_arch_mat='run/fromLeastereo/sceneflow/best/architecture/matching_network_path.npy' \
                # --cell_arch_mat='run/fromLeastereo/sceneflow/best/architecture/matching_genotype.npy' \