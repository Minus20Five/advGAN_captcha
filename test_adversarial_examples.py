import torch
import torchvision.datasets
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader
from advgan import models
from solver import captcha_setting, one_hot_encoding
from solver.captcha_cnn_model import CNN
from solver.my_dataset import get_train_data_loader, get_test_data_loader
from torchvision.utils import save_image

use_cuda=True
image_nc=1
batch_size = 128
logging_batch = 100
test_on_training = False

def main():
    gen_input_nc = image_nc

    # Define what device we are using
    # print("CUDA Available: ",torch.cuda.is_available())
    # device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    device = 'cpu'

    # load the pretrained model
    pretrained_model = "./models/model.pkl"
    target_model = CNN().to(device)
    target_model.load_state_dict(torch.load(pretrained_model))
    target_model.eval()

    # load the generator of adversarial examples
    pretrained_generator_path = './models/netG_epoch_35.pth'
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()

    # test adversarial examples in MNIST training dataset
    if test_on_training:
        print('Testing on training set...')
        train_dataloader = get_train_data_loader(batch_size=1)
        num_correct = 0
        for i, data in enumerate(train_dataloader, 0):
            test_img, test_label = data
            perturbation = pretrained_G(test_img)
            perturbation = torch.clamp(perturbation, -0.3, 0.3)
            adv_img = perturbation + test_img
            adv_img = torch.clamp(adv_img, 0, 1)
            predict_label = target_model(adv_img)

            c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
            true_label = one_hot_encoding.decode(test_label.numpy()[0])

            num_correct += 1 if predict_label == true_label else 0

            if (i+1) % logging_batch == 0:
                print('\tTotal num_correct for first {} images: {}'.format(i+1, num_correct))
                print('\tAccuracy for first {} images: {}'.format(i+1, num_correct/(i+1)))

        print('Total num_correct: ', num_correct)
        print('Total accuracy of adv imgs in training set: %f\n'%(num_correct/len(train_dataloader)))

    # test adversarial examples in MNIST testing dataset
    print('Testing on training set...')
    test_dataloader = get_test_data_loader()
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        predict_label = target_model(adv_img)

        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        true_label = one_hot_encoding.decode(test_label.numpy()[0])

        num_correct += 1 if predict_label == true_label else 0

        save_image(test_img[0], '.\{}-{}.png'.format(i, 'original'))
        save_image(adv_img[0], '.\{}-{}.png'.format(i, 'adv'))

        if (i+1) % logging_batch == 0:
            print('\tTotal num_correct for first {} images: {}'.format(i+1, num_correct))
            print('\tAccuracy for first {} images: {}'.format(i+1, num_correct/(i+1)))

    print('Total num_correct: ', num_correct)
    print('Total accuracy of adv imgs in testing set: %f\n'%(num_correct/len(test_dataloader)))

if __name__ == '__main__':
    main()