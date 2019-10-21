import torch
import argparse

import captcha_setting
from captcha_cnn_model import CNN
from captcha_general import predict_n_batches
# from utils.utils import training_device

parser = argparse.ArgumentParser(description='Test the CAPTCHA solver')

parser.add_argument(
    '--dir', '-d',
    help='the folder the test and training data is in',
    type=str,
    # assumes running from project root and not in this file's directory (i.e python ./main.py)
    default='data'
)

def test_captcha_solver(args):
    cnn = CNN()
    cnn.load_state_dict(torch.load(captcha_setting.get_model_save_name(args.dir), map_location='cuda'))  #
    cnn.eval()
    print("load cnn net.")
    predict_n_batches(model=cnn, n=20, dir=args.dir)


if __name__ == '__main__':
    args = parser.parse_args()
    test_captcha_solver(args)