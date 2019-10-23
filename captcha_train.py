# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable

from solver import my_dataset, captcha_setting
from solver.captcha_cnn_model import CNN
from advgan.models import Generator

# Hyper Parameters
num_epochs = 30
batch_size = 100
learning_rate = 0.001

clamp = 0.05

def main():

    print("CUDA Available: ", torch.cuda.is_available())

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = "cuda:0"
    else:
        device = "cpu"

    cnn = CNN().to(device)
    generator = Generator(1, 1).to(device)
    generator.load_state_dict(torch.load('./models/generator-5epoch-pt05clamp.pkl', map_location=device))
    generator.eval()
    cnn.train()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)



    # Train the Model
    train_dataloader = my_dataset.get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            images_var = Variable(images)
            labels = Variable(labels.float())
            predict_labels = cnn(images_var)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels) * 0.7
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            perturbations = generator(images)
            perturbations = torch.clamp(perturbations, 0.0-clamp, clamp)
            adv_images = perturbations + images
            adv_images = torch.clamp(adv_images, 0, 1)
            
            perturbed_images_var = Variable(adv_images)
            predict_labels_perturbed = cnn(perturbed_images_var)
            loss = criterion(predict_labels_perturbed, labels) * 0.3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
        torch.save(cnn.state_dict(), captcha_setting.MODEL_PATH + '/solver-adv-epoch' + str(epoch) + '.pkl')   #current is model.pkl
        print("saved model")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())

    torch.save(cnn.state_dict(), captcha_setting.MODEL_PATH + '/solver-adv.pkl')   #current is solver-adv.pkl
    print("save last model")

if __name__ == '__main__':
    main()


