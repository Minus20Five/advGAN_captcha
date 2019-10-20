import os
import random
import string
from multiprocessing import Process

from wheezy.captcha import image as wheezy_captcha

NUM_THREADS = 4  # number of threads to use in parallel captcha generation
TRAIN_SIZE = 8 # size of training set
TEST_SIZE = 1  # size of test set
LABEL_SEQ_LENGTH = 4  # number of characters in captcha
LABEL_SEQ_VALUE = string.digits + string.ascii_uppercase

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FONTS = ["Consolas.ttf", "DroidSansMono.ttf","ComicSans.ttf"]
BG_COLORS = ["#ffffff", "#e57373", "#212121"]
GREY_SCALE = ["#FAFAFA","#E0E0E0", "#9E9E9E", "#616161", "#212121"]

class WheezyCaptchaStyle:
    def __init__(self, warp=(0.27,0.21), 
                       rotate_angle=25, 
                       offset=(0.1,0.2), 
                       bg_color="#ffffff", 
                       font="Consolas.ttf",
                       font_color="#5C87B2",
                       font_squeeze=0.8,
                       curve=None,
                       noise_color=None,
                       noise_level=None,
                       noise_number=None):
        self.warp = wheezy_captcha.warp(dx_factor=warp[0],dy_factor=warp[1])
        self.rotate = wheezy_captcha.rotate(angle=rotate_angle)
        self.offset = wheezy_captcha.offset(dx_factor=offset[0], dy_factor=offset[1])
        self.bg_color = wheezy_captcha.background(color=bg_color)
        self.font = [os.path.join(FILE_DIR, "../font", font)]
        self.font_color = font_color
        self.squeeze = font_squeeze
        self.curve = None
        self.noise = None
        if curve is not None:
            self.curve = wheezy_captcha.curve(number=curve, color=font_color)
        if noise_color is not None:
            self.noise = wheezy_captcha.noise(color=noise_color, number=noise_number)


class WheezyCaptcha:
    """Create an image CAPTCHA with wheezy.captcha."""

    def __init__(self, width=160, height=60, fonts=None):
        self._width = width
        self._height = height


    def generate_image(self, chars, style):
        text_drawings = [
            style.warp,
            style.rotate,
            style.offset,
        ]

        drawings = [ 
            style.bg_color,
            wheezy_captcha.text(fonts=style.font, 
                                drawings=text_drawings, 
                                squeeze_factor=style.squeeze,
                                color=style.font_color),
            style.curve,
            style.noise,
            wheezy_captcha.smooth(),
        ]

        drawings_param = [ i for i in drawings if i is not None ]
        fn = wheezy_captcha.captcha(
            drawings=drawings_param,
            width=self._width,
            height=self._height,
        )
        return fn(chars)


def generate_image(path, start, end, style=None):
    image_captcha = WheezyCaptcha()
    captcha_style = WheezyCaptchaStyle() if style is None else style
    for idx in range(start, end):
        label_seq = ''.join(random.choices(LABEL_SEQ_VALUE, k=LABEL_SEQ_LENGTH))
        newImage = image_captcha.generate_image(label_seq, captcha_style)
        newImage.convert("L") # converts to greyscale (somehow)
        newImage.save(os.path.join(path, label_seq + '.png'))


def get_batchsize(total_num, groups):
    bsz = total_num // groups
    if total_num % groups:
        bsz += 1
    return bsz

def generate_data(dir_name, style=None):
    dir_path = os.path.join(FILE_DIR, dir_name)
    # if os.path.exists(dir_path):
    #     raise Exception("Data folder already exists. Remove folder before re-running this script")
    for phase in ['test', 'train']:
        path = os.path.join(dir_path, phase)

        if not os.path.exists(path):
            os.makedirs(path)

        dataset_size = TEST_SIZE if phase == 'test' else TRAIN_SIZE
        threads = []
        batch_size = get_batchsize(dataset_size, NUM_THREADS)

        for t in range(NUM_THREADS):
            start, end = t * batch_size, (t + 1) * batch_size
            if t == NUM_THREADS - 1:
                end = dataset_size
            if style is None:
                p = Process(target=generate_image, args=(path, start, end))
            else:
                p = Process(target=generate_image, args=(path, start, end, style))
            p.start()
            threads.append(p)

        for p in threads:
            p.join()
        print(phase + ' done.')


if __name__ == '__main__':

    # Old function to generate default data
    # generate_default_data("data")

    # New functions for generating different styles\
    CAP_STYLES = {}
    CAP_STYLES["Minimal_distortion_red_bg"] = WheezyCaptchaStyle(warp=(0,0), rotate_angle=0, offset=(0.2,0.2), bg_color=BG_COLORS[1], font=DEFAULT_FONTS[1]) 
    # CAP_STYLES["Sans_more_rotate"] = WheezyCaptchaStyle(warp=(0.2,0.2), rotate_angle=35, offset=(0.2,0.2), bg_color=BG_COLORS[0], font=DEFAULT_FONTS[2],
    #                                  font_squeeze=0.75)
    # CAP_STYLES["More_Warp"] = WheezyCaptchaStyle(warp=(0.4,0.3))
    # CAP_STYLES["Varying_Contrast_1"] = WheezyCaptchaStyle(rotate_angle=15, bg_color=GREY_SCALE[0], font=DEFAULT_FONTS[0],curve=4)
    # CAP_STYLES["Varying_Contrast_2"] = WheezyCaptchaStyle(rotate_angle=15, bg_color=GREY_SCALE[1], font=DEFAULT_FONTS[0],curve=3)
    # CAP_STYLES["Varying_Contrast_3"] = WheezyCaptchaStyle(rotate_angle=15, bg_color=GREY_SCALE[2], font=DEFAULT_FONTS[0],curve=2)
    # CAP_STYLES["Varying_Contrast_4"] = WheezyCaptchaStyle(rotate_angle=15, bg_color=GREY_SCALE[3], font=DEFAULT_FONTS[0],curve=10)
    # CAP_STYLES["Varying_Contrast_5"] = WheezyCaptchaStyle(rotate_angle=15, bg_color=GREY_SCALE[4], font=DEFAULT_FONTS[0],curve=8)
                                     
    # generate_data("style_1", CAP_STYLE_1)
    # generate_data("style_2", CAP_STYLE_2)
    
    for i, (name, style) in enumerate(CAP_STYLES.items()):
        generate_data(name,style)
