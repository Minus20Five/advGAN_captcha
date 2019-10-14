import torch

from advgan.advGAN import AdvGAN_Attack
from solver import captcha_setting, one_hot_encoding
from solver.captcha_cnn_model import CNN

from utils.utils import training_device

if __name__ == '__main__':
    solver = CNN()
    solver.load_state_dict(torch.load(captcha_setting.SOLVER_SAVE_PATH, map_location=training_device()))
    solver.eval()
    advGan = AdvGAN_Attack(model=solver)
    advGan.load_models()

    num_attacked, num_correct = advGan.attack_n_batches(n=100, save_images=False)
    print("Total: {} Correct: {} Accuracy: {}".format(num_attacked, num_correct, num_correct/num_attacked))