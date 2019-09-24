#coding=utf-8

import os
import time
import numpy as np
import torch
import sys
import pdb
sys.path.append("/home/wenwens/Downloads/semantic_align_gan_v9")

from collections import OrderedDict
from options.augment_options import AugmentOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.util import tensor2im, parsing2im, label_2_onhot
from torch.autograd import Variable
from val import get_valList

opt = AugmentOptions().parse()
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

opt.stage = 5
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt, opt.which_G)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter    
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    # count = 0
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == 0
        b_label_show_tensor = data['b_label_show_tensor']
        
        infer = save_fake
        model.train()
        aug_losses, fake_b_parsing = model.module.forward_augment(data)
        losses, fake_b_parsing = model.module.forward_target(data)
        
        print(model.module.netG.model[-2].bias)
        print(model.module.skeleton_net.alpha)
        print(model.module.skeleton_net.alpha.grad)

        ### ['G_GAN', 'G_GAN_Feat', 'G_L1', 'D_real', 'D_fake']
        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_real'] + loss_dict['D_fake']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_L1']

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

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == 0:
            errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ############## Display output images and Val images ######################
        # if save_fake:
            # val_list = get_valList(model, opt)
            # train_list = [('b_label', tensor2im(b_label_show_tensor[0])),
            #                ('a_parsing', parsing2im(label_2_onhot(a_parsing_tensor[0], parsing_label_nc=opt.parsing_label_nc))),
            #                ('b_parsing', parsing2im(label_2_onhot(b_parsing_tensor[0], parsing_label_nc=opt.parsing_label_nc))),
            #                ('fake_b_parsing', parsing2im(fake_b_parsing.data[0]))]
            # val_list[0:0] = train_list
            # val_list = train_list
            # visuals = OrderedDict(val_list)

            # visualizer.display_current_results(visuals, epoch, total_steps)


        ### save latest model
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
       
    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    # if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
    #     model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()