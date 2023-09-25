import glob
import sys
import cv2
# from skimage.measure import compare_psnr
from torch import nn

sys.path.append('../')
import os
import torch.nn.functional
import numpy as np
import torch.optim as optim
from option import args
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tool.MyDataSet import Art_nosie_Dataset, GenDataset
from tool.MyDataSet import Real_Dataset
from tool.common_tools import ModelTrainer
from tool.common_tools import save_to_file
from tool.common_tools import batch_PSNR
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from image_similarity_measures.quality_metrics import fsim, psnr, ssim
from tqdm import tqdm
import random

if args.n_colors == 3:
    color = 'color'

    if args.sigma == 200.0:
        print("Use the realnoise dataset...........")
        My_Test_Dataset = Real_Dataset
        My_Train_Dataset = Real_Dataset

        args.train_dataset = "real_dataset/realworld/"
        args.test_dataset = "real_dataset"
        print("n_color is 3, Training in real_dastaset, evaluate in realdataset")

    else:
        My_Test_Dataset = Art_nosie_Dataset
        My_Train_Dataset = GenDataset

        args.train_dataset = "CBSD432+pristine+DIV2k+Flick2k"

        if args.test_dataset == "CBSD68":
            print("n_color is 3, Training in {}, evaluate in {}".format(args.train_dataset,
                                                                                 args.test_dataset))
        else:
            print("n_color is 3, Training in CBSD432, evaluate in {}".format(args.test_dataset))

elif args.n_colors == 1:
    color = 'g'
    args.train_dataset = "BSD432+pristine+DIV2K+Flick2k"

    My_Test_Dataset = Art_nosie_Dataset
    My_Train_Dataset = Art_nosie_Dataset

    if args.test_dataset == "CBSD68":
        args.test_dataset = "BSD68"
        print("n_color is 1, Training in {}, evaluate in {}".format(args.train_dataset, args.test_dataset))
    else:
        print("n_color is 1, Training in BSD432, evaluate in {}".format(args.test_dataset))
else:
    raise ValueError("args.n_color must equal 1 or 3 in interage")


train_dir = args.dataset_dir_base + os.path.join('train', args.train_dataset)
test_dir = args.dataset_dir_base + os.path.join('test', args.test_dataset)
save_model_dir = args.save_base + os.path.join(args.dir_model)
save_state_dir = args.save_base + os.path.join(args.dir_state)
save_loss_dir = args.save_base + os.path.join(args.dir_loss)
save_test_dir = args.save_base + os.path.join(args.dir_test_img)

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_id


def log(*args, **kwargs):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def test(args, data_loader, save_test_dir, save=False, model_file=None, loss_f=None):
    args.mode = 'test'

    if save:
        torch.manual_seed(1234)

    import model
    _model = model.Model(args, model_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log('load trained model to GPU')

    psnrs = []
    ssims = []
    fsims = []
    L = []
    save_dict = {}
    psnr_dict = {}
    ssim_dict = {}

    for data in tqdm(data_loader):

        ori_img, nos_img, img_name = data

        img_name = img_name[0]
        noise = torch.FloatTensor(ori_img.size()).normal_(mean=0, std=args.sigma / 255.)


        with torch.no_grad():
            nos_img = nos_img.to(device)

            if args.model_name == 'ffdnet':
                sigma = torch.full((1, 1, 1, 1), args.sigma * args.rgb_range / 255.).type_as(nos_img)
                output = _model(nos_img, sigma)
            else:
                output = _model(nos_img)

        output = output.cpu()
        nos_img = nos_img.cpu()

        output = output.squeeze_()
        ori_img = ori_img.squeeze_()
        nos_img = nos_img.squeeze_()

        loss_ = loss_f(output, ori_img) / 2.0

        output = output.mul_(255. / args.rgb_range)
        ori_img = ori_img.mul_(255. / args.rgb_range)
        nos_img = nos_img.mul_(255. / args.rgb_range)

        if args.n_colors == 3:
            output = output.permute(1, 2, 0)        # chw --> hwc
            ori_img = ori_img.permute(1, 2, 0)      # chw --> hwc
            nos_img = nos_img.permute(1, 2, 0)      # chw --> hwc


        np_output = np.uint8(output.detach().clamp(0, 255).round().numpy())         # tensor --> np(intenger)
        np_img_rgb = np.uint8(ori_img.detach().clamp(0, 255).round().numpy())       # tensor --> np(intenger)
        np_img_nos = np.uint8(nos_img.detach().clamp(0, 255).round().numpy())       # tensor --> np(intenger)

        if save:
            img_save = np.hstack((np_img_rgb, np_output, np_img_nos))

            save_dict[img_name] = img_save
            save_dict['gt_'+img_name] = np_img_rgb
            save_dict['out_'+img_name] = np_output
            save_dict['noise_'+img_name] = np_img_nos

        psnr_x_ = peak_signal_noise_ratio(np_output, np_img_rgb)
        if args.n_colors == 3:
            ssim_x_ = structural_similarity(np_img_rgb, np_output, channel_axis=2)
        else:
            ssim_x_ = structural_similarity(np_img_rgb, np_output)

        if args.n_colors == 3:
            fsim_x_ = fsim(np_img_rgb, np_output)
        else:
            re_img = np.transpose(np.repeat(np.expand_dims(np_img_rgb, 0), 3, 0), (1, 2, 0))
            re_out = np.transpose(np.repeat(np.expand_dims(np_output, 0), 3, 0), (1, 2, 0))
            fsim_x_ = fsim(re_img, re_out)

        psnr_dict[img_name] = psnr_x_
        ssim_dict[img_name] = ssim_x_

        psnrs.append(psnr_x_)
        ssims.append(ssim_x_)
        fsims.append(fsim_x_)
        L.append(loss_.item())

    psnr_avg = np.mean(psnrs)
    ssim_avg = np.mean(ssims)
    fsim_avg = np.mean(fsims)
    loss_avg = np.mean(L)

    psnr_max = np.max(psnrs)
    ssim_max = np.max(ssims)

    if save:
        now_time = datetime.now()
        time_str = datetime.strftime(now_time, '%m-%d_%H-%M')

        save_path = os.path.join(save_test_dir, args.model_name, color, str(args.sigma),args.test_dataset)

        if not os.path.exists(save_path):   # 没有该文件夹，则创建该文件夹
            os.makedirs(save_path)
            print("Make the dir:{}".format(save_path))

        txtname = os.path.join(save_path, 'test_result.txt')
        if not os.path.exists(txtname):
            os.system(r"touch {}".format(txtname))

        save_to_file(os.path.join(save_path, 'test_result.txt'),
                     "Time: {}, psnr_avg: {:.4f},  ssim_avg: {:.4f}\npsnr_max: {:.4f}, ssim_max:{:.4f} \nfsim_avg:{:.4f}\n".\
                     format(time_str, psnr_avg, ssim_avg, psnr_max, ssim_max, fsim_avg))

        p_str = " ".join(sys.argv)
        save_to_file(os.path.join(save_path, 'test_result.txt'), p_str+'\n')

        for k1, k2 in zip(psnr_dict, ssim_dict):
            save_to_file(os.path.join(save_path, 'test_result.txt'),
                         "{}, psnr: {}, ssim: {}\n".format(k1, psnr_dict[k1], ssim_dict[k2]))

        for k in save_dict:
            img = cv2.cvtColor(save_dict[k], cv2.COLOR_RGB2BGR)  # RGB->BGR
            cv2.imwrite(os.path.join(save_path, k), img)

    print("Psnr_Avg:{:.2f}, ssim_avg:{:.4f}\nPsnr_Max: {:.2f}, ssim_max:{:.4f} \nFsim_avg:{:.4f}".\
          format(psnr_avg, ssim_avg, psnr_max, ssim_max, fsim_avg))

    return psnr_avg, ssim_avg, loss_avg


def train(args):
    # ---------------------------------------configuration---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = args.batch_size

    # ====================================step 1/5data========================================================
    # 构建MyDataset实例
    test_data = My_Test_Dataset(args, data_dir=test_dir, mode='test')

    # 构建DataLoder
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # ===================================step2/5 model========================================================
    import model
    _model = model.Model(args)

    # ====================================step3/5 Loss_function================================================
    if args.loss_func.lower() == 'l2':
        from model_common.loss import sum_squared_error
        # criterion = sum_squared_error()
        criterion = nn.MSELoss(reduction='sum')
    elif args.loss_func.lower() == 'ssim':
        from model_common.loss import MSSSIM
        criterion = MSSSIM()
    elif args.loss_func.lower() == 'l1':
        criterion = nn.L1Loss(reduction='sum')
    elif args.loss_func.lower() == 'smooth':
        criterion = nn.SmoothL1Loss(reduction='sum')
    else:
        raise ValueError("Please input the correct loss function with --loss_func $loss function(mse or ssim)....")
    
    # ====================================step4/5 优化器========================================================
    optimizer = optim.Adam(_model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    if args.start_epoch > 0:
        print("Start to load state from %d epoch.............." % args.start_epoch)

        state_path = os.path.join(save_state_dir, args.model_name, color, str(args.sigma),
                                  'state_%03d_sigma%d.t7' % (args.start_epoch, args.sigma))
        checkpoint = torch.load(state_path)
        _model.model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    milestone = [15-args.start_epoch,
                 22-args.start_epoch,
                 24-args.start_epoch,
                 26-args.start_epoch,
                 28-args.start_epoch,
                 30-args.start_epoch,
                 31-args.start_epoch,
                 190]

    milestone = list(filter(lambda x: x > 0, milestone))

    scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.5)  # learning rates

    if args.flag == 0:
        if not os.path.exists(os.path.join(save_loss_dir, args.model_name, color, str(args.sigma))):
            os.makedirs(os.path.join(save_loss_dir, args.model_name, color, str(args.sigma)))
            f = open(os.path.join(save_loss_dir, args.model_name, color, str(args.sigma), 'train_result.txt'), 'w')
            f.close()
            print("Make the dir: {}".format(
                os.path.join(save_loss_dir, args.model_name, color, str(args.sigma), 'train_result.txt')))
        p_str = " ".join(sys.argv)
        save_to_file(os.path.join(save_loss_dir, args.model_name, color, str(args.sigma), 'train_result.txt'), p_str+"\n")

    # ====================================step5/5 trianing......===============================================
    for epoch in range(start_epoch, args.epochs):
        train_data = My_Train_Dataset(args, data_dir=train_dir, mode='train')
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=1)
        epoch_loss = ModelTrainer.train(train_loader, _model, criterion, optimizer, epoch, device, args.epochs)

        print("===============Epoch[{:0>3}/{:0>3}]  Train loss:{:.4f}  LR:{}=================".format(
            epoch+1, args.epochs, epoch_loss, optimizer.param_groups[0]["lr"]))
        scheduler.step()

        # ======================================save=============================================================
        if args.flag == 0:
            if epoch % 1 == 0:
                if not os.path.exists(os.path.join(save_model_dir, args.model_name, color,str(args.sigma))):
                    os.makedirs(os.path.join(save_model_dir, args.model_name, color, str(args.sigma)))
                    print("Make the dir: {}".format(os.path.join(save_model_dir, args.model_name, color, str(args.sigma))))

                torch.save(_model.model.state_dict(), os.path.join(save_model_dir, args.model_name, color, str(args.sigma), 'model_%03d_sigma%d.pth' % (epoch+1, args.sigma)))

                if not os.path.exists(os.path.join(save_state_dir, args.model_name, color, str(args.sigma))):
                    os.makedirs(os.path.join(save_state_dir, args.model_name, color, str(args.sigma)))
                    print("Make the dir: {}".format(os.path.join(save_state_dir, args.model_name, color, str(args.sigma))))

                state = {
                    'epoch': epoch+1,
                    'net': _model.model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                torch.save(state, os.path.join(save_state_dir, args.model_name, color, str(args.sigma), 'state_%03d_sigma%d.t7' % (epoch+1, args.sigma)))

            now_time = datetime.now()
            time_str = datetime.strftime(now_time, '%m-%d-%H:%M:%S')
            print(time_str)

            psnr_avg, ssim_avg, _loss = test(args,
                                             test_loader,
                                             save_test_dir,
                                             save=False,
                                             model_file=_model,
                                             loss_f=criterion)

            if not os.path.exists(os.path.join(save_loss_dir, args.model_name, color, str(args.sigma))):
                os.makedirs(os.path.join(save_loss_dir, args.model_name, color, str(args.sigma)))
                f = open(os.path.join(save_loss_dir, args.model_name, color, str(args.sigma), 'train_result.txt'), 'w')
                f.close()
                print("Make the dir: {}".format(os.path.join(save_loss_dir, args.model_name, color, str(args.sigma), 'train_result.txt')))

            save_to_file(os.path.join(save_loss_dir, args.model_name, color, str(args.sigma), 'train_result.txt'),
                         "\nTime: {}, Epoch: {},  Loss: {:.4f}, psnr: {:.4f},  ssim: {:.4f}, test_loss: {:.4f}"\
                         .format(time_str, epoch+1, epoch_loss, psnr_avg, ssim_avg, _loss))


if __name__ == '__main__':
    if args.mode == 'train':
        print("Start to train.......")
        train(args)
    elif args.mode == 'test':
        print("Start to test.......")

        test_data = My_Test_Dataset(args, data_dir=test_dir, mode='test')
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

        criterion = nn.MSELoss(reduction='sum')
        ppsnr_avg, ssim_avg, _loss = test(args,
                                      test_loader,
                                      save_test_dir,
                                      save=True,
                                      model_file=args.model_file_name,
                                      loss_f=criterion)

    elif args.mode == 'inference':
        print("Start to inference.......")
        pass