# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from solver import my_dataset
from solver.captcha_cnn_model import CNN

# Hyper Parameters
num_epochs = 30
batch_size = 100
learning_rate = 0.001

def main():

    print("CUDA Available: ", torch.cuda.is_available())

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = "cuda:0"
    else:
        device = "cpu"

    cnn = CNN().to(device)
    cnn.train()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)



    # Train the Model
    train_dataloader = my_dataset.get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            images = Variable(images)
            labels = Variable(labels.float())
            predict_labels = cnn(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i, "loss:", loss.item())
            if (i+1) % 100 == 0:
                torch.save(cnn.state_dict(), "./model.pkl")   #current is model.pkl
                print("save model")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
    torch.save(cnn.state_dict(), "./model.pkl")   #current is model.pkl
    print("save last model")

if __name__ == '__main__':
    main()


