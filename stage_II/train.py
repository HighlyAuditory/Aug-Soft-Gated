#coding=utf-8

import os
import time
import numpy as np
import torch
import sys
# sys.path.append("/home/disk2/donghaoye/ACMMM/semantic_align_gan_v9")

from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.util import tensor2im, parsingim_2_tensor
from torch.autograd import Variable
from val import get_valList

opt = TrainOptions().parse()

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 100

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
print('#num_iterations_per_epoch = %d' % opt.num_iterations_per_epoch)

model = create_model(opt)
visualizer = Visualizer(opt)

# total_steps = (start_epoch-1) * dataset_size + epoch_iter
total_steps = (start_epoch-1) * (opt.num_iterations_per_epoch * opt.batchSize) + epoch_iter
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        #epoch_iter = epoch_iter % dataset_size
        epoch_iter = epoch_iter % (opt.num_iterations_per_epoch * opt.batchSize)
        # iterations 是指多少个batchSize，epoch_iter指的是多少个图片。

    count = 0
    for i, data in enumerate(dataset, start=epoch_iter):
        if count >= opt.num_iterations_per_epoch:
            count = 0
            break
        count = count + 1

        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == 0

        a_image_tensor = data['a_image_tensor']         # 3
        b_image_tensor = data['b_image_tensor']         # 3
        b_label_tensor = data['b_label_tensor']         # 18
        a_parsing_tensor = data['a_parsing_tensor']     # 1
        b_parsing_tensor = data['b_parsing_tensor']     # 1
        b_label_show_tensor = data['b_label_show_tensor']
        theta_aff = data['theta_aff_tensor']            # 2
        theta_tps = data['theta_tps_tensor']            # 2
        theta_aff_tps = data['theta_aff_tps_tensor']    # 2
        a_jpg_path = data['a_jpg_path']
        b_jpg_path = data['b_jpg_path']

        input_tensor = torch.cat([a_image_tensor, b_image_tensor, b_label_tensor, a_parsing_tensor, b_parsing_tensor, \
                                  theta_aff, theta_tps, theta_aff_tps], dim=1)
        input_var = Variable(input_tensor.type(torch.cuda.FloatTensor))

        infer = save_fake
        model.train()
        losses, fake_b = model.forward(input_var, infer)

        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        loss_D = (loss_dict['D_real'] + loss_dict['D_fake']) * 0.5
        loss_G = loss_dict['G_GAN']  + loss_dict['G_L1'] + loss_dict['G_VGG'] + loss_dict['G_TV'] + loss_dict['G_GAN_Feat'] + loss_dict['G_Parsing']

        ############### Backward Pass ####################
        # update generator weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()

        # update discriminator weights
        if not opt.no_GAN_loss:
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()

        if infer:
            a_parsing_rgb_tensor = parsingim_2_tensor(a_parsing_tensor[0], opt=opt, parsing_label_nc=opt.parsing_label_nc)
            b_parsing_rgb_tensor = parsingim_2_tensor(b_parsing_tensor[0], opt=opt, parsing_label_nc=opt.parsing_label_nc)

            show_image_tensor_1 = torch.cat((a_image_tensor, b_label_show_tensor, b_image_tensor), dim=3)
            show_image_tensor_2 = torch.cat((a_parsing_rgb_tensor, b_parsing_rgb_tensor, fake_b.data[0:1, :, :, :].cpu()), dim=3)
            show_image_tensor = torch.cat((show_image_tensor_1[0:1, :, :, :], show_image_tensor_2), dim=2)

        ############## Display results and errors ##########
        if total_steps % opt.print_freq == 0:
            errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ############## Display output images and Val images ######################
        if save_fake:
            val_list = get_valList(model, opt, data_loader.dataset)
            train_list = [('a-b-fake_b', tensor2im(show_image_tensor[0]))]
            val_list[0:0] = train_list
            # val_list = train_list
            visuals = OrderedDict(val_list)
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest models
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest models (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save models for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the models at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    # if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
    #     model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()



