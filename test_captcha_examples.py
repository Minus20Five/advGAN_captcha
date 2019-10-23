import torch

from solver import captcha_setting
from solver.captcha_cnn_model import CNN
from solver.captcha_general import predict_n_batches
from utils.utils import training_device

if __name__ == '__main__':
    cnn = CNN()
    cnn.load_state_dict(torch.load(captcha_setting.MODEL_PATH + '/solver-adv-epoch1.pkl', map_location=training_device()))  #
    cnn.eval()
    print("load cnn net.")
    predict_n_batches(model=cnn, n=20)