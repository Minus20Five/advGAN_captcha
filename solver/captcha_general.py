import numpy as np
import torch
from torch.autograd import Variable

from solver import captcha_setting, one_hot_encoding, my_dataset
from solver.captcha_cnn_model import CNN
from solver.captcha_setting import SOLVER_SAVE_PATH
from utils.utils import training_device


def decode_captcha_out(prediction):
    c0 = captcha_setting.ALL_CHAR_SET[
        np.argmax(prediction[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
        prediction[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
        prediction[0,
        2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
        prediction[0,
        3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    return '%s%s%s%s' % (c0, c1, c2, c3)


def predict_n_batches(model, n=1):
    test_dataloader = my_dataset.get_test_data_loader()

    correct = 0
    total = 0
    batches_predicted = 0
    for i, (images, labels) in enumerate(test_dataloader):
        image = images
        vimage = Variable(image)
        batches_predicted += 1
        predict_label = decode_captcha_out(model(vimage))
        true_label = one_hot_encoding.decode(labels.numpy()[0])
        total += labels.size(0)
        if predict_label == true_label:
            correct += 1
        if total%200==0:
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
        if batches_predicted >= n:
            break
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))


