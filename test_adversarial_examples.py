import torch
import argparse

from advgan.advGAN import AdvGAN_Attack
from solver import captcha_setting, one_hot_encoding, my_dataset
from solver.captcha_cnn_model import CNN
from solver.captcha_general import decode_captcha_batch
from utils.utils import mkdir_p, training_device

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
parser.add_argument(
    '--adversarial', '-a',
    help='test the effects of an adversarial attack with generator specified with -g, or the default generator',
    action='store_true'
)

args = parser.parse_args()
target_solver_path = args.target
generator_path = args.generator
save_images = args.save
batches = args.batches
adversarial = args.adversarial

if __name__ == '__main__':
    print('Saving is: {}'.format('on' if save_images else 'off'))
    print('Testing on: {}'.format('adversarial data' if adversarial else 'original data'))
    solver = CNN()
    solver.load_state_dict(torch.load(target_solver_path, map_location=training_device()))
    solver.eval()

    print('Attacking {} batches'.format(batches))

    if adversarial:
        advGan = AdvGAN_Attack(model=solver, device='cpu')
        advGan.load_models(generator_filename=generator_path)

        num_attacked, num_correct = advGan.attack_n_batches(n=batches, save_images=save_images)
        print("Total: {} Correct: {} Accuracy: {}".format(num_attacked, num_correct, num_correct/num_attacked))
    else:
        test_dataloader = my_dataset.get_test_data_loader(batch_size=batches)
        times_attacked = 0
        num_attacked = 0
        num_correct = 0

        for i, data in enumerate(test_dataloader, 0):
            times_attacked += 1
            test_images, test_labels = data
            num_attacked += test_images.shape[0]  # the first dimension of data is the batch_size
            
            predict_labels = decode_captcha_batch(solver(test_images))
            true_labels = [one_hot_encoding.decode(test_label) for test_label in test_labels.numpy()]
            for predict_label, true_label in zip(predict_labels, true_labels):
                num_correct += 1 if predict_label == true_label else 0

            print("Total: {} Correct: {} Accuracy: {}".format(num_attacked, num_correct, num_correct / num_attacked))
