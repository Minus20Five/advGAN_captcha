from wheezy.captcha import image as wheezy_captcha
import os
import random
import string
from multiprocessing import Process

NUM_THREADS = 4  # number of threads to use in parallel captcha generation
TRAIN_SIZE = 80000  # size of training set
TEST_SIZE = 10000  # size of test set
LABEL_SEQ_LENGTH = 4  # number of characters in captcha
LABEL_SEQ_VALUE = string.digits + string.ascii_uppercase

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_FONTS = [os.path.join(FILE_DIR, "font", "Consolas.ttf")]


class WheezyCaptcha:
    """Create an image CAPTCHA with wheezy.captcha."""

    def __init__(self, width=160, height=60, fonts=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS

    def generate_image(self, chars):
        text_drawings = [
            wheezy_captcha.warp(),
            wheezy_captcha.rotate(),
            wheezy_captcha.offset(),
        ]
        fn = wheezy_captcha.captcha(
            drawings=[
                wheezy_captcha.background(color="#ffffff"),
                wheezy_captcha.text(fonts=self._fonts, drawings=text_drawings),
                # wheezy_captcha.curve(number=5),
                # wheezy_captcha.noise(color="#dddddd"),
                wheezy_captcha.smooth(),
            ],
            width=self._width,
            height=self._height,
        )
        return fn(chars)


def generate_image(path, start, end):
    image_captcha = WheezyCaptcha()
    for idx in range(start, end):
        label_seq = ''.join(random.choices(LABEL_SEQ_VALUE, k=LABEL_SEQ_LENGTH))
        newImage = image_captcha.generate_image(label_seq)
        newImage.convert("L") # converts to greyscale (somehow)
        newImage.save(os.path.join(path, label_seq + '.png'))


def get_batchsize(total_num, groups):
    bsz = total_num // groups
    if total_num % groups:
        bsz += 1
    return bsz


if __name__ == '__main__':
    dir_path = os.path.join(FILE_DIR, "data")
    if os.path.exists(dir_path):
        raise Exception("Data folder already exists. Remove folder before re-running this script")
    for phase in ['test', 'train']:
        path = os.path.join(dir_path, phase)
        os.makedirs(path)

        dataset_size = TEST_SIZE if phase == 'test' else TRAIN_SIZE
        threads = []
        batch_size = get_batchsize(dataset_size, NUM_THREADS)

        for t in range(NUM_THREADS):
            start, end = t * batch_size, (t + 1) * batch_size
            if t == NUM_THREADS - 1:
                end = dataset_size
            p = Process(target=generate_image, args=(path, start, end))
            p.start()
            threads.append(p)

        for p in threads:
            p.join()
        print(phase + ' done.')
