import torch
import argparse

from advgan.advGAN import AdvGAN_Attack
from solver import captcha_setting
from solver.captcha_cnn_model import CNN
from solver.lstm.lstm import StackedLSTM
from utils.utils import training_device

parser = argparse.ArgumentParser(description='Train the AdvGan against captchas with a solver')
parser.add_argument(
    '--target', '-t',
    help='the path of the target solver we would like to generate noise to defeat',
    type=str,
    # assumes running from project root and not in this file's directory (i.e python ./test_adversarial_examples.py)
    default=captcha_setting.SOLVER_SAVE_PATH
)
parser.add_argument(
    '--generator', '-g',
    help='the path of the adversarial noise generator we would like to use to generate noise',
    type=str,
    # assumes running from project root and not in this file's directory (i.e python ./test_adversarial_examples.py)
    default=captcha_setting.GENERATOR_FILE_PATH
)
parser.add_argument(
    '--save', '-s',
    help='saves the images being tested, as well as the noise both seperately and applied onto original image',
    action='store_true'
)
parser.add_argument(
    '--batches', '-b',
    help='number of batches to test',
    type=int,
    default=captcha_setting.BATCH_SIZE
)

args = parser.parse_args()
target_solver_path = args.target
generator_path = args.generator
save_images = args.save
batches = args.batches

if __name__ == '__main__':
    print('Saving is: {}'.format('on' if save_images else 'off'))
    # solver = CNN()
    # solver.load_state_dict(torch.load(captcha_setting.SOLVER_SAVE_PATH, map_location=training_device()))
    # solver.eval()
    solver = StackedLSTM()
    solver.load_state_dict(torch.load('lstm_batch_5.pkl', map_location=device))
    solver.eval()
    h = solver.init_hidden(64)  # in
    advGan = AdvGAN_Attack(model=solver, device='cpu')
    advGan.load_models(generator_filename=generator_path)

    print('Attacking {} batches'.format(batches))
    num_attacked, num_correct = advGan.attack_n_batches(n=batches, save_images=save_images)
    print("Total: {} Correct: {} Accuracy: {}".format(num_attacked, num_correct, num_correct/num_attacked))