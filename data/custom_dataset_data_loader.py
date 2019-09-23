import torch.utils.data
from data.base_data_loader import BaseDataLoader
import pdb

def CreateDataset(opt):
    dataset = None

    if opt.stage == 1:
        from data.stage_I_dataset import Stage_I_Dataset
        dataset = Stage_I_Dataset()
    elif opt.stage == 2:
        from data.stage_II_dataset import Stage_II_Dataset
        dataset = Stage_II_Dataset()
    elif opt.stage == 3:
        from data.w_o_stage_I_dataset import W_O_Stage_I_Dataset
        dataset = W_O_Stage_I_Dataset()
    elif opt.stage == 11:
        from data.stage_I_skeleton_dataset import Stage_I_Skeleton_Dataset
        dataset = Stage_I_Skeleton_Dataset()
    elif opt.stage == 12:
        from data.stage_I_II_dataset import Stage_I_II_Dataset
        dataset = Stage_I_II_Dataset()
    elif opt.stage == 123:
        from data.stage_I_II_dataset_penn import Stage_I_II_Dataset_Penn
        dataset = Stage_I_II_Dataset_Penn()
    elif opt.stage == 4:
        from data.augment_dataset import Augment_Dataset
        dataset = Augment_Dataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
