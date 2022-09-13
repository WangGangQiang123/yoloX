import os

import torch


def save(self, apath, epoch, is_best=False):
    target = self.get_model()
    torch.save(
        target.state_dict(),
        os.path.join(apath, 'model', 'G:\yolox-pytorch-main\yolox-pytorch-main\model_data/best_epoch_weights.pth'),
        _use_new_zipfile_serialization=False
    )
    if is_best:
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'G:\yolox-pytorch-main\yolox-pytorch-main\model_data/best_epoch_weights01.pth'),
            _use_new_zipfile_serialization=False
        )
