# -*- coding: UTF-8 -*-
import os
# 验证码中的字符
# string.digits + string.ascii_uppercase
from os import path

def get_test_path(dirname='data'):
    return dirname + os.path.sep + 'test'

def get_train_path(dirname='data'):
    return dirname + os.path.sep + 'train'

def get_model_save_name(dirname='data'):
    return path.join(MODEL_PATH, "model_{}.pkl".format(dirname))

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

BATCH_SIZE = 64

TRAIN_DATASET_PATH = get_train_path()
TEST_DATASET_PATH = get_test_path()

MODEL_PATH = "models"
SOLVER_FILE_NAME = "model.pkl"
GENERATOR_FILE_NAME = "generator.pkl"
DISCRIMINATOR_FILE_NAME = "discriminator.pkl"

SOLVER_SAVE_PATH = path.join(MODEL_PATH, SOLVER_FILE_NAME)
GENERATOR_FILE_PATH = path.join(MODEL_PATH, GENERATOR_FILE_NAME)
DISCRIMINATOR_FILE_PATH = path.join(MODEL_PATH, DISCRIMINATOR_FILE_NAME)

IMAGE_PATH = path.join(MODEL_PATH, "images")


