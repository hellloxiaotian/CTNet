import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as P
import torch.utils.model_zoo
from importlib import import_module


class Model(nn.Module):
    def __init__(self, args, model=None):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.patch_size = args.patch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_GPUs = args.n_GPUs
        self.mode = args.mode
        self.save_models = args.save_models

        if model is None or isinstance(model, str):
            module = import_module('model.' + args.model_name.lower())
            self.model, ours = module.make_model(args)
            if ours == 0:
                model = 0
        else:
            self.model = model
            print("Model is Created!")

        if self.mode == 'train':
            self.model.train()

            if self.args.pretrain != '':
                self.model.load_state_dict(
                    torch.load(
                        os.path.join(self.args.dir_model, 'pre_train', self.args.model_name, self.args.pre_train)),
                    strict=False)

            self.model = nn.DataParallel(self.model.to(self.device), device_ids=[i for i in range(self.n_GPUs)])
        elif self.mode == 'test':

            if isinstance(model, str):
                dict_path = model
                print("Be ready to load model from {}".format(dict_path))

                load_dict = torch.load(dict_path)

                try:
                    self.model.load_state_dict(load_dict, strict=True)
                except RuntimeError:
                    from collections import OrderedDict
                    new_dict = OrderedDict()

                    for key, _ in load_dict.items():    # 去掉开头module.前缀
                        new_dict[key[7:]] = load_dict[key]

                    self.model.load_state_dict(new_dict, strict=True)

            self.model = nn.DataParallel(self.model.to(self.device), device_ids=[i for i in range(self.n_GPUs)])

            self.model.eval()

    def forward(self, x, sigma=None):
        if self.mode == 'train':
            return self.model(x)
        elif self.mode == 'test':
            if self.args.num_layers == 0:
                if sigma is None:
                    return self.model(x)
                else:
                    return self.model(x, sigma)
            else:
                return self.forward_chop(x)
        else:
            raise ValueError("Choose the train or test model......")

    def forward_chop(self, x, shave=12):
        x.cpu()
        batchsize = self.args.crop_batch_size
        h, w = x.size()[-2:]
        padsize = int(self.patch_size)
        shave = int(self.patch_size / 2)

        h_cut = (h - padsize) % (int(shave / 2))
        w_cut = (w - padsize) % (int(shave / 2))

        x_unfold = F.unfold(x, padsize, stride=int(shave / 2)).transpose(0, 2).contiguous()

        x_hw_cut = x[..., (h - padsize):, (w - padsize):]
        y_hw_cut = self.model.forward(x_hw_cut.cuda()).cpu()

        x_h_cut = x[..., (h - padsize):, :]
        x_w_cut = x[..., :, (w - padsize):]
        y_h_cut = self.cut_h(x_h_cut, w, w_cut, padsize, shave, batchsize)
        y_w_cut = self.cut_w(x_w_cut, h, h_cut, padsize, shave, batchsize)

        x_h_top = x[..., :padsize, :]
        x_w_top = x[..., :, :padsize]
        y_h_top = self.cut_h(x_h_top, w, w_cut, padsize, shave, batchsize)
        y_w_top = self.cut_w(x_w_top, h, h_cut, padsize, shave, batchsize)

        x_unfold = x_unfold.view(x_unfold.size(0), -1, padsize, padsize)
        y_unfold = []

        x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
        x_unfold.cuda()
        for i in range(x_range):
            y_unfold.append(
                self.model(x_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu()
            )

        y_unfold = torch.cat(y_unfold, dim=0)

        y = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                                     ((h - h_cut), (w - w_cut)), padsize,
                                     stride=int(shave / 2))

        y[..., :padsize, :] = y_h_top
        y[..., :, :padsize] = y_w_top

        y_unfold = y_unfold[..., int(shave / 2):padsize - int(shave / 2), int(shave / 2):padsize - int(shave / 2)].contiguous()
        y_inter = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                                           ((h - h_cut - shave), (w - w_cut - shave)),
                                           padsize - shave, stride=int(shave / 2))

        y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, padsize - shave, stride=int(shave / 2)),
            ((h - h_cut - shave), (w - w_cut - shave)), padsize - shave,
            stride=int(shave / 2))

        y_inter = y_inter / divisor

        y[..., int(shave / 2 ):(h - h_cut) - int(shave / 2),
        int(shave / 2):(w - w_cut) - int(shave / 2)] = y_inter

        y = torch.cat([y[..., :y.size(2) - int((padsize - h_cut) / 2), :],
                       y_h_cut[..., int((padsize - h_cut) / 2 + 0.5):, :]], dim=2)
        y_w_cat = torch.cat([y_w_cut[..., :y_w_cut.size(2) - int((padsize - h_cut) / 2), :],
                             y_hw_cut[..., int((padsize - h_cut) / 2 + 0.5):, :]], dim=2)
        y = torch.cat([y[..., :, :y.size(3) - int((padsize - w_cut) / 2)],
                       y_w_cat[..., :, int((padsize - w_cut) / 2 + 0.5):]], dim=3)
        return y.cuda()

    def cut_h(self, x_h_cut, w, w_cut, padsize, shave, batchsize):

        x_h_cut_unfold = F.unfold(x_h_cut, padsize, stride=int(shave / 2)).transpose(0, 2).contiguous()

        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0), -1, padsize, padsize)
        x_range = x_h_cut_unfold.size(0) // batchsize + (x_h_cut_unfold.size(0) % batchsize != 0)
        y_h_cut_unfold = []
        x_h_cut_unfold.cuda()
        for i in range(x_range):
            y_h_cut_unfold.append(
                self.model(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu()
            )
        y_h_cut_unfold = torch.cat(y_h_cut_unfold, dim=0)

        y_h_cut = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize, (w - w_cut)), padsize, stride=int(shave / 2))
        y_h_cut_unfold = y_h_cut_unfold[..., :,
                         int(shave / 2 ):padsize - int(shave / 2 )].contiguous()
        y_h_cut_inter = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize, (w - w_cut - shave)), (padsize , padsize - shave),
            stride=int(shave / 2))

        y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, (padsize, padsize - shave),
                                       stride=int(shave / 2)), (padsize, (w - w_cut - shave)),
            (padsize, padsize - shave), stride=int(shave / 2))
        y_h_cut_inter = y_h_cut_inter / divisor

        y_h_cut[..., :, int(shave / 2):(w - w_cut) - int(shave / 2)] = y_h_cut_inter

        return y_h_cut

    def cut_w(self, x_w_cut, h, h_cut, padsize, shave, batchsize):

        x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave / 2)).transpose(0, 2).contiguous()

        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0), -1, padsize, padsize)
        x_range = x_w_cut_unfold.size(0) // batchsize + (x_w_cut_unfold.size(0) % batchsize != 0)
        y_w_cut_unfold = []
        x_w_cut_unfold.cuda()
        for i in range(x_range):
            # y_w_cut_unfold.append(P.data_parallel(self.model, x_w_cut_unfold[i * batchsize:(i + 1) * batchsize, ...], range(self.n_GPUs)).cpu())
            y_w_cut_unfold.append(self.model(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu())
        y_w_cut_unfold = torch.cat(y_w_cut_unfold, dim=0)

        y_w_cut = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut), padsize), padsize, stride=int(shave / 2))
        y_w_cut_unfold = y_w_cut_unfold[..., int(shave / 2):padsize - int(shave / 2), :].contiguous()
        y_w_cut_inter = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut - shave), padsize), (padsize - shave, padsize),
            stride=int(shave / 2))

        y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, (padsize - shave, padsize), stride=int(shave / 2)), ((h - h_cut - shave), padsize),
            (padsize - shave, padsize), stride=int(shave / 2))
        y_w_cut_inter = y_w_cut_inter / divisor

        y_w_cut[..., int(shave / 2):(h - h_cut) - int(shave / 2), :] = y_w_cut_inter

        return y_w_cut