import torch
from advgan.advGAN import AdvGAN_Attack
from solver.captcha_cnn_model import CNN
from solver import captcha_setting

from solver.my_dataset import get_train_data_loader


def main():
    use_cuda = True
    image_nc = 1  # 'nc' means number of channels ( i think)
    epochs = 40
    batch_size = 128

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # pretrained_model = "./MNIST_target_model.pth"
    # targeted_model = MNIST_target_net().to(device)
    targeted_model = CNN().to(device)
    targeted_model.load_state_dict(torch.load(captcha_setting.SOLVER_SAVE_PATH, map_location=device))
    targeted_model.eval()
    # model_num_labels = 10
    model_num_labels = captcha_setting.ALL_CHAR_SET_LEN

    # MNIST train dataset and dataloader declaration
    # mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    # dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader = get_train_data_loader()
    advGAN = AdvGAN_Attack(targeted_model,
                           device=device,
                           image_nc=image_nc)
    advGAN.train(dataloader, epochs)
    advGAN.save_models()


if __name__ == '__main__':
    main()
