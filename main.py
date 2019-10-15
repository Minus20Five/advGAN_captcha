import torch

from advgan.advGAN import AdvGAN_Attack
from solver import captcha_setting
from solver.captcha_cnn_model import CNN
from solver.my_dataset import get_train_data_loader

from utils.utils import training_device

def main():
    use_cuda = True
    image_nc = 1  # 'nc' means number of channels ( i think)
    epochs = 1
    batch_size = 128

    # pretrained_model = "./MNIST_target_model.pth"
    # targeted_model = MNIST_target_net().to(device)
    targeted_model = CNN()
    targeted_model.load_state_dict(torch.load(captcha_setting.SOLVER_SAVE_PATH, map_location=training_device()))
    targeted_model.eval()
    # model_num_labels = 10
    model_num_labels = captcha_setting.ALL_CHAR_SET_LEN

    # MNIST train dataset and dataloader declaration
    # mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    # dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader = get_train_data_loader()
    advGAN = AdvGAN_Attack(targeted_model,
                           image_nc=image_nc)
    advGAN.train(dataloader, epochs)
    advGAN.save_models()


if __name__ == '__main__':
    main()
