import torch

from solver import captcha_setting
from solver.captcha_cnn_model import CNN
from solver.captcha_general import predict_n_batches
from utils.utils import training_device

if __name__ == '__main__':
    cnn = CNN()
    cnn.load_state_dict(torch.load(captcha_setting.SOLVER_SAVE_PATH, map_location=training_device()))  #
    cnn.eval()
    print("load cnn net.")
    predict_n_batches(model=cnn)