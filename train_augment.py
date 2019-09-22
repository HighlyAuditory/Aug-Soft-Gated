import sys
import pdb
import torch

from options.augment_options import AugmentOptions
from data.data_loader import CreateDataLoader
from models.augment_model import AugmentModel

sys.path.append("~/Downloads/semantic_align_gan_v9/models") 

opt = AugmentOptions().parse(save=False)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 100

opt.stage = 4 ## choose augment_dataset.py, will load one side and new skeletons
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

aug_model = AugmentModel()
aug_model.initialize(opt)
start_epoch = 0
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    
    for i, data in enumerate(dataset):
        with torch.autograd.set_detect_anomaly(True):
            # save_fake = total_steps % opt.display_freq == 0
            # infer = save_fake
            infer = 0
            print(i)
            aug_model.train()
            aug_prediction, aug_reparsing_loss = aug_model.forward_aug(data, infer)
            # aug_model.optimizer_G.zero_grad()
            aug_reparsing_loss.backward()
            aug_model.optimizer_G.step()
            print(aug_model.net_SK.alpha.grad)
            # pdb.set_trace()
            for n,d in aug_model.main_model.netG.named_parameters():
                if n == 'up.13.bias':
                    print(d)
        
            # b_prediction, reparsing_loss = aug_model.forward_target(data, infer)
            # # update augment model
            # aug_model.optimizer_SK.zero_grad()
            # reparsing_loss.backward()
            # aug_model.optimizer_SK.step()
        