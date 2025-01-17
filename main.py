import torch
import argparse

from os import path
from advgan.advGAN import AdvGAN_Attack
from solver import captcha_setting
from solver.captcha_cnn_model import CNN
from solver.my_dataset import get_train_data_loader

from utils.utils import training_device

parser = argparse.ArgumentParser(description='Train the AdvGan against captchas with a solver')
parser.add_argument(
    '--target', '-t',
    help='the path of the target solver we would like to generate noise to defeat',
    type=str,
    # assumes running from project root and not in this file's directory (i.e python ./main.py)
    default=captcha_setting.SOLVER_SAVE_PATH
)
parser.add_argument(
    '--epochs', '-e',
    help='number of epochs to train for',
    type=int,
    default=40
)
parser.add_argument(
    '--cuda', '-c',
    help='train on cuda',
    action='store_true'
)
parser.add_argument(
    '--bounds', '-b',
    help='amount to bound the noise generation',
    type=float,
    default=0.05
)
parser.add_argument(
    '--generator_name', '-g',
    help='name to save generator as under the default models folder',
    default=captcha_setting.GENERATOR_FILE_NAME
)
parser.add_argument(
    '--discriminator_name', '-d',
    help='name to save discriminator as under the default models folder',
    default=captcha_setting.DISCRIMINATOR_FILE_NAME
)
parser.add_argument(
    '--smooth', '-s',
    help='apply label smoothing for training the AdvGan',
    action='store_true'
)

args = parser.parse_args()
target_solver_path = args.target
device = 'cuda' if args.cuda else 'cpu'
epochs = args.epochs
bounds = args.bounds
generator_path = path.join(captcha_setting.MODEL_PATH, args.generator_name)
discriminator_path = path.join(captcha_setting.MODEL_PATH, args.discriminator_name)
smooth = args.smooth

image_nc = 1  # 'nc' means number of channels ( i think)
batch_size = 128

def main():
    print('Label smoothing is {}'.format('ON' if smooth else 'OFF'))
    print('Training for {} epochs with clamp amount +/- {}...'.format(epochs, bounds))
    # pretrained_model = "./MNIST_target_model.pth"
    # targeted_model = MNIST_target_net().to(device)
    targeted_model = CNN()
    targeted_model.load_state_dict(torch.load(target_solver_path, map_location=training_device(device=device)))
    targeted_model.eval()
    # model_num_labels = 10
    model_num_labels = captcha_setting.ALL_CHAR_SET_LEN

    # MNIST train dataset and dataloader declaration
    # mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    # dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader = get_train_data_loader()
    advGAN = AdvGAN_Attack(targeted_model,
                            device=device,
                            image_nc=image_nc,
                            clamp=bounds)
    advGAN.train(dataloader, epochs, smooth=smooth)
    advGAN.save_models(generator_filename=generator_path, discriminator_filename=discriminator_path)


if __name__ == '__main__':
    main()
