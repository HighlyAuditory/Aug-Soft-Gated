CUDA_VISIBLE_DEVICES=0  python ./rebuttal/test_I_II_multi_pose_penn.py --name I_II --dataroot ./ --pairs_path ./rebuttal/penn_image_list_20180730.txt --stage 123 --gpu_ids=0 --parsing_label_nc 20 --pose_file_path ./rebuttal/penn_pose_sequence_20180730.txt --how_many 50

CUDA_VISIBLE_DEVICES=0  python ./rebuttal/test_I_II.py --name I_II --dataroot ~/Downloads/semantic_align_gan_v9/datasets/deepfashion/paper_images/256/JOINT/deepfasion/00013/id_00000390 --stage 12 --gpu_ids=0 --parsing_label_nc 20 --how_many 80 --pairs_path ./datasets/deepfashion/In-shop_AB_HD_p1024/Anno/my_test_stage_I.txt

CUDA_VISIBLE_DEVICES=0  python ./rebuttal/test_I_II_multi_pose_penn.py --name I_II --dataroot ./ --pairs_path ./rebuttal/penn_image_list_20180730.txt --stage 123 --gpu_ids=0 --parsing_label_nc 20 --pose_file_path ./rebuttal/penn_pose_sequence_20180730.txt --how_many 50 --resize_or_crop resize_and_crop | rescale_with

# stage I
CUDA_VISIBLE_DEVICES=0 python ./stage_I/train.py --name stage_I_gan_ganFeat_noL1_oneD_Parsing_bz50_parsing20_20190409_debug --dataroot /home/wenwens/Datasets/DeepFashion/paper_img --batchSize 2 --stage 1 --tf_log --no_L1_loss --num_D 1 --no_TV_loss  --no_flip --gpu_ids=0 --parsing_label_nc 20

# stage II (original batch size is 12)
CUDA_VISIBLE_DEVICES=0  python ./stage_II/train.py --name gan_L1_feat_vgg_notv_noparsing_afftps_05091451 --dataroot /home/wenwens/Datasets/DeepFashion/paper_img --batchSize 2 --stage 2 --tf_log --gpu_ids=0 --no_flip --which_G wapResNet_v3_afftps --parsing_label_nc 20 --no_TV_loss --num_D 1 --no_Parsing_loss --pairs_path ./datasets/deepfashion/In-shop_AB_HD_p1024/Anno/my_train_pairs_deepfashion_stage_II.txt

# Soft-Gated
# test stage I and II
CUDA_VISIBLE_DEVICES=0  python ./rebuttal/test_I_II.py --name I_II --dataroot /home/wenwens/Datasets/DeepFashion/paper_img  --stage 12 --gpu_ids=0 --parsing_label_nc 20 --how_many 80
# test stage I
CUDA_VISIBLE_DEVICE=0 python ./stage_I/test_all_parsing_label.py --name stage_I_gan_ganFeat_noL1_oneD_Parsing_bz50_parsing20_04222 --dataroot /home/wenwens/Datasets/DeepFashion/paper_img --stage 1 --which_img all --which_epoch 100 --parsing_label_nc 20 --how_many 10000000 --phase test
# stage I tested with paper imgs, results in ./results/stage_I_gan_ganFeat_noL1_oneD_Parsing_bz50_parsing20_04222/test_100/images

# test stage II
CUDA_VISIBLE_DEVICES=0 python ./stage_II/test_all.py --name gan_L1_feat_vgg_notv_noparsing_afftps_05102228 --dataroot /home/wenwens/Datasets/DeepFashion/paper_img --stage 2 --gpu_ids=0 --which_G wapResNet_v3_afftps --which_img all --which_epoch 40 --parsing_label_nc 20 --how_many 80
# stage II tested with paper imgs, results in ./results/gan_L1_feat_vgg_notv_noparsing_afftps_05102228/test_40/images

# stage I training
CUDA_VISIBLE_DEVICES=0,1 python ./stage_I/train.py --name stage_I_gan_ganFeat_noL1_oneD_Parsing --dataroot /data/haoye/Img --batchSize 50 --stage 1 --tf_log --no_L1_loss --num_D 1 --no_TV_loss --no_flip --gpu_ids=0,1 --phase test

# stage II training
CUDA_VISIBLE_DEVICES=0,1 python ./stage_II/train.py --name gan_L1_feat_vgg_notv_noparsing_affine --dataroot /data/haoye/Img --batchSize 12 --stage 2 --tf_log --gpu_ids=0,1 --no_flip --which_G wapResNet_v3_affps --parsing_label_nc 20 --no_TV_loss --num_D 1 --no_Parsing_loss --num_iterations_per_epoch 5000 --continue_train

# 根据parsing的RGB图，利用GEOCNN生成变形的参数theta
generate_theta_json_20channels.py

# parts of the paper
In our method, we combine affine and TPS to obtain the transformation mapping through a siamesed convolutional neural network following GEO [25]. To be specific, we first estimate the affine transformation between the condition and synthesized parsing. Based on the results from affine estimation, we then estimate TPS transformation parameters between warping results from the affine transformation and target parsing.
human parsing parser
Ke Gong, Xiaodan Liang, Xiaohui Shen, and Liang Lin. Look into person: Self-supervised structure-sensitive learning and a new benchmark for human parsing. In CVPR, pages 6757– 6765,

pose estimator
Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Realtime multi-person 2d pose estimation using part affinity fields. In CVPR, 2017.

the affine [7]/TPS [2, 25] (Thin-Plate Spline) transformation estimator
affine
Ping Dong and Nikolas P Galatsanos. Affine transformation resistant watermarking based on image normalization. In ICIP (3), pages 489–492, 2002.
TPS
Fred L. Bookstein. Principal warps: Thin-plate splines and the decomposition of deformations. IEEE Transactions on pattern analysis and machine intelligence, 11(6):567–585, 1989.
