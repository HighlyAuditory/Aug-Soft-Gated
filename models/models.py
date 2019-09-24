### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import pdb

def create_model(opt, name):
    if opt.stage == 1:
        from .stage_I_model import Stage_I_Model
        model = Stage_I_Model()
    elif opt.stage == 2:
        from .semantic_align_model import SemanticAlignModel
        model = SemanticAlignModel()
    elif opt.stage == 3:
        from .w_o_semantic_align_model import W_O_SemanticAlignModel
        model = W_O_SemanticAlignModel()
    elif opt.stage == 11:
        from .skeleton_stage_I_model import Skeleton_Stage_I_Model
        model = Skeleton_Stage_I_Model()
    elif opt.stage == 5:
        from .augment_stage_I_model import Augment_Stage_I_Model
        model = Augment_Stage_I_Model()
    else:
        print ("create model ERROR!!!")

    model.initialize(opt, name)

    print("models [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
