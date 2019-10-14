# -*- coding: UTF-8 -*-
import math

import numpy as np
import torch
from torch.autograd import Variable
#from visdom import Visdom # pip install Visdom
from solver import captcha_setting, my_dataset
from solver.captcha_cnn_model import CNN


def predict_n(n=math.inf):
    cnn = CNN()
    cnn.load_state_dict(torch.load(captcha_setting.SOLVER_SAVE_PATH))
    cnn.eval()
    print("load cnn net.")

    predict_dataloader = my_dataset.get_predict_data_loader()

    #vis = Visdom()
    count = 0
    for i, (images, labels) in enumerate(predict_dataloader):
        count += 1
        image = images
        vimage = Variable(image)
        predict_label = cnn(vimage)

        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

        c = '%s%s%s%s' % (c0, c1, c2, c3)
        print(c)
        #vis.images(image, opts=dict(caption=c))
        if count >= n:
            break


if __name__ == '__main__':
    predict_n(1)


