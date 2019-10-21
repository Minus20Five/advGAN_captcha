import torch
import argparse

from solver import captcha_setting
from solver.captcha_cnn_model import CNN
from solver.captcha_general import predict_n_batches
from utils.utils import training_device

parser = argparse.ArgumentParser(description='Test the CAPTCHA solver')

parser.add_argument(
    '--dir', '-d',
    help='the folder the test and training data is in',
    type=str,
    # assumes running from project root and not in this file's directory (i.e python ./main.py)
    default='data'
)

args = parser.parse_args()

def test_captcha_solver(args):
    cnn = CNN()
    cnn.load_state_dict(torch.load(captcha_setting.SOLVER_SAVE_PATH, map_location=training_device()))  #
    cnn.eval()
    print("load cnn net.")
    predict_n_batches(model=cnn, n=20, dir=args.dir)


if __name__ == '__main__':
    test_captcha_solver(args)