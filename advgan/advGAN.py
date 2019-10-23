import os
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from advgan import models
from solver import captcha_setting, one_hot_encoding
from solver.captcha_general import decode_captcha_batch
from solver.my_dataset import get_test_data_loader
from utils.utils import mkdir_p, training_device

models_path = './models/'

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 model,
                 image_nc=1,
                 box_min=0,
                 box_max=1,
                 device='cuda',
                 clamp=0.05):
        self.device = training_device(device)
        print("Using: " + self.device)

        self.model = model.to(self.device)
        self.input_nc = image_nc
        self.gen_input_nc = image_nc
        self.box_min = box_min
        self.box_max = box_max
        self.clamp = clamp
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(self.device)
        self.netDisc = models.Discriminator(image_nc).to(self.device)
        self.batch_size = 64
        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        self.dir = captcha_setting.MODEL_PATH
        mkdir_p(self.dir)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def load_models(self, generator_filename=captcha_setting.GENERATOR_FILE_PATH,
                    discriminator_filename=captcha_setting.DISCRIMINATOR_FILE_PATH):
        self.netG.load_state_dict(torch.load(generator_filename, map_location=self.device))
        self.netG.to(self.device)
        print('Generator sucessfully loaded from {}'.format(generator_filename))
        self.netDisc.load_state_dict(
            torch.load(os.path.join(discriminator_filename), map_location=self.device))
        self.netDisc.to(self.device)
        print('Discriminator sucessfully loaded from {}'.format(discriminator_filename))

    def save_models(self, generator_filename=captcha_setting.GENERATOR_FILE_PATH,
                    discriminator_filename=captcha_setting.DISCRIMINATOR_FILE_PATH):
        torch.save(self.netG.state_dict(), generator_filename)
        print('Generator sucessfully saved at {}'.format(generator_filename))
        torch.save(self.netDisc.state_dict(), discriminator_filename)        
        print('Discriminator sucessfully saved at {}'.format(discriminator_filename))

    # using pretrained solver and advGAN generator, generate noise for one CATPCHA image,
    # save the image, noise, and noise + image
    def attack_n_batches(self, n=1, save_images=False, quiet=False):
        self.model.eval()

        pretrained_G = self.netG
        pretrained_G.eval()

        test_dataloader = get_test_data_loader(self.batch_size)
        times_attacked = 0
        num_attacked = 0
        num_correct = 0
        mkdir_p(captcha_setting.IMAGE_PATH)
        for i, data in enumerate(test_dataloader, 0):
            times_attacked += 1
            test_images, test_labels = data
            num_attacked += test_images.shape[0]  # the first dimension of data is the batch_size
            perturbations = pretrained_G(test_images)
            perturbations = torch.clamp(perturbations, 0.0-self.clamp, self.clamp)
            adv_images = perturbations + test_images
            adv_images = torch.clamp(adv_images, 0, 1)

            predict_labels = decode_captcha_batch(self.model(adv_images))
            true_labels = [one_hot_encoding.decode(test_label) for test_label in test_labels.numpy()]
            for predict_label, true_label in zip(predict_labels, true_labels):
                num_correct += 1 if predict_label == true_label else 0

            if save_images:
                for j in range(len(test_images)):
                    test_image = test_images[j]
                    perturbation_image = perturbations[j]
                    adv_image = adv_images[j]
                    save_image(test_image,
                               path.join(captcha_setting.IMAGE_PATH, 'batch-{}-{}-{}.png'.format(i, j, 'original')))
                    save_image(perturbation_image,
                               path.join(captcha_setting.IMAGE_PATH, 'batch-{}-{}-{}.png'.format(i, j, 'noise')))
                    save_image(adv_image,
                               path.join(captcha_setting.IMAGE_PATH, 'batch-{}-{}-{}.png'.format(i, j, 'adv')))

            if times_attacked >= n:
                break

            if not quiet:
                print("\tTotal: {} Correct: {} Accuracy: {}".format(num_attacked, num_correct, num_correct / num_attacked))

        return num_attacked, num_correct

    def train_batch(self, x, labels, smooth=False):
        self.netG.train()
        self.netDisc.train()
        # optimize D
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, 0.0-self.clamp, self.clamp) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            real_output_tensor = (torch.rand_like(pred_real, device=self.device) * 0.3 ) + 0.7 if smooth else torch.ones_like(pred_real, device=self.device)
            
            loss_D_real = F.mse_loss(pred_real, real_output_tensor)
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            fake_output_tensor = torch.rand_like(pred_fake, device=self.device) * 0.3 if smooth else torch.zeros_like(pred_fake, device=self.device)
            
            loss_D_fake = F.mse_loss(pred_real, fake_output_tensor)
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            logits_model = self.model(adv_images)
            loss_adv = -F.multilabel_soft_margin_loss(logits_model, labels.float())
            # probs_model = F.softmax(logits_model, dim=1)
            # onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            # C&W loss function
            # real = torch.sum(onehot_labels * probs_model, dim=1)
            # other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            # zeros = torch.zeros_like(other)
            # loss_adv = torch.max(real - other, zeros)
            # loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs, smooth=False):
        self.netG.train()
        self.netDisc.train()
        for epoch in range(1, epochs + 1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, labels, smooth=smooth)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

                if (i % 20 == 0):
                    print("\tstep %d:\n\tloss_D: %.3f, loss_G_fake: %.3f,\
                        \n\tloss_perturb: %.3f, loss_adv: %.3f, \n" %
                          (i, loss_D_batch, loss_G_fake_batch,
                           loss_perturb_batch, loss_adv_batch))
            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum / num_batch, loss_G_fake_sum / num_batch,
                   loss_perturb_sum / num_batch, loss_adv_sum / num_batch))

            # save generator
            if epoch % 20 == 0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pkl'
                torch.save(self.netG.state_dict(), netG_file_name)
