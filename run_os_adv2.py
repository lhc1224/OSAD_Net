import argparse
import os
from OSADv2.dataset import get_loader
from OSADv2.solver import Solver


import torch
import numpy as np
import  random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(12)

def main(config):
    if config.mode == 'train':
        train_loader, dataset = get_loader(batch_size=config.batch_size,
                                           image_size=config.image_size,
                                           img_root=config.image_root,
                                           mask_root=config.mask_root,
                                           ref=config.ref_path,
                                           txts_path=config.txts_path,
                                           lst_path=config.train_lst_path,
                                           mode='train',root_path=None,
                                           txt_path=None,num_thread=config.num_thread,
                                           test_ref_root=None)
        train = Solver(train_loader, None, config)
        train.train(save_path=config.save_path)
    elif config.mode == 'test':
       # model_paths=os.listdir(config.model)

        test_loader, dataset = get_loader(batch_size=1,
                                          image_size=config.image_size,
                                          img_root=None,
                                          mask_root=None,
                                          ref=None,
                                          txts_path=None,
                                          lst_path=None,
                                          mode='test',
                                          root_path=config.test_image_path,
                                          txt_path=config.txt_path,
                                          num_thread=config.num_thread,
                                          test_ref_root=config.test_ref_root)
        #for model_path in model_paths:
        #    m_path=os.path.join()

        models_paths=os.listdir(config.model_root)
        for path in models_paths:
            if not path[-3:]=="pth":
                continue
            config.model=os.path.join(config.model_root,path)
            config.save_image_path=os.path.join(config.model_root,path[:-9])
            print(config.model)
            print(config.save_image_path)
            test = Solver(None, test_loader, config)
            test.test(save_path=config.save_image_path,image_root=config.test_image_path)
    else:
        raise IOError("illegal input!!!")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)

    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument("--in_channels",type=int,default=256)
    parser.add_argument("--output_channels", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="resnet")


    # Training settings
    parser.add_argument('--epoch', type=int, default=42)  # 12, now x3
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--load_bone', type=str, default='')
    parser.add_argument('--epoch_save', type=int, default=1)  # 2, now x3
    parser.add_argument('--pre_trained', type=str, default=None)
    parser.add_argument('--k',type=int,default=64)
    parser.add_argument('--image_root',type=str,default='datasets/PADv2/divide_1/train/images/')
    parser.add_argument('--mask_root',type=str,default="datasets/PADv2/divide_1/train/masks/")
    parser.add_argument('--ref_path',type=str,default="datasets/PADv2/divide_1/train/refs/")
    parser.add_argument('--txts_path', type=str, default="datasets/PADv2/divide_1/train/txts/")
    parser.add_argument("--train_lst_path",type=str,default="datasets/PADv2/divide_1/train/train.lst")
    parser.add_argument("--save_path",type=str,default="OSADv2/save_models/")
    parser.add_argument("--num_GPU",type=int,default=1)

    # Testing settings
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument("--model_root",type=str,default="OSADv2/save_models/")
    parser.add_argument('--image_size', type=str, default=320)
    parser.add_argument('--n_layers', type=int, default=50)
    parser.add_argument('--test_image_path',type=str,default="datasets/PADv2/divide_1/test/images")
    parser.add_argument('--txt_path',type=str,default="datasets/PADv2/divide_1/test/test_ref_3.txt")
    parser.add_argument('--save_image_path',type=str,default=None)
    parser.add_argument("--test_ref_root",type=str,default="datasets/PADv2/divide_1/test/refs/")
    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    config = parser.parse_args()

    main(config)

