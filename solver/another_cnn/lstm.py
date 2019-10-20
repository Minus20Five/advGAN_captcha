# CAPTCHA model found at https://github.com/denkuzin/captcha_solver
from itertools import groupby
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.dataset import random_split

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from captcha.image import ImageCaptcha

BLANK_LABEL = 10
BATCH_SIZE = 64
IMAGE_WIDTH = 160

#
# image = ImageCaptcha()
#
# for chars in range(0, 10000):
#     image.write(f'{chars:>04}', f'{chars:>04}.png')
from utils.utils import training_device


class CaptchaDataset(Dataset):
    """CAPTCHA dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = list(Path(root_dir).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])

        if self.transform:
            image = self.transform(image)

        label_sequence = [int(c) for c in self.image_paths[index].stem]
        return (image, torch.tensor(label_sequence))

    def __len__(self):
        return len(self.image_paths)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# dataset = CaptchaDataset(root_dir='data/captcha', transform=transform)
#
# dataloader = DataLoader(dataset, batch_size=10000)
#
# for batch_index, (inputs, labels) in enumerate(dataloader):
#     print(f'Mean: {inputs.mean()}, Variance: {inputs.std()}')


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.89165,), (0.14776,)),
])

# dataset = CaptchaDataset(root_dir='data/captcha', transform=transform)

# train_dataset, test_dataset = random_split(dataset, [128*64, 28*64])  # total images: 9984

# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
train_dataloader = DataLoader(CaptchaDataset(root_dir='data/captcha/train', transform=transform), batch_size=64, shuffle=True)


class StackedLSTM(nn.Module):
    def __init__(self, input_size=60, output_size=11, hidden_size=512, num_layers=2):
        super(StackedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout()
        self.fc = nn.Linear(hidden_size, output_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, inputs, hidden):
        # batch_size, seq_len, input_size = inputs.shape
        outputs, hidden = self.lstm(inputs, hidden)
        outputs = self.dropout(outputs)
        outputs = torch.stack([self.fc(outputs[i]) for i in range(IMAGE_WIDTH)])
        outputs = F.log_softmax(outputs, dim=2)
        return outputs, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_size).zero_())


device = training_device()
net = StackedLSTM().to(device)

criterion = nn.CTCLoss(blank=10)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


# Train
net.train()  # set network to training phase

epochs = 1
# for each pass of the training dataset
# for epoch in range(epochs):
#     train_loss, train_correct, train_total = 0, 0, 0
#
#     h = net.init_hidden(BATCH_SIZE)
#
#     # for each batch of training examples
#     for batch_index, (inputs, targets) in enumerate(train_dataloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         h = tuple([each.data for each in h])
#
#         batch_size, channels, height, width = inputs.shape
#
#         # reshape inputs: NxCxHxW -> WxNx(HxC)
#         inputs = (inputs
#                   .permute(3, 0, 2, 1)
#                   .contiguous()
#                   .view((width, batch_size, -1)))
#
#         optimizer.zero_grad()  # zero the parameter gradients
#         outputs, h = net(inputs, h)  # forward pass
#
#         # compare output with ground truth
#         input_lengths = torch.IntTensor(batch_size).fill_(width)
#         target_lengths = torch.IntTensor([len(t) for t in targets])
#         loss = criterion(outputs, targets, input_lengths, target_lengths)
#
#         loss.backward()  # backpropagation
#         nn.utils.clip_grad_norm_(net.parameters(), 10)  # clip gradients
#         optimizer.step()  # update network weights
#
#         # record statistics
#         prob, max_index = torch.max(outputs, dim=2)
#         train_loss += loss.item()
#         train_total += len(targets)
#
#         for i in range(batch_size):
#             raw_pred = list(max_index[:, i].cpu().numpy())
#             pred = [c for c, _ in groupby(raw_pred) if c != BLANK_LABEL]
#             target = list(targets[i].cpu().numpy())
#             # print('Pred: {} Target: {}'.format(pred, target))
#             if pred == target:
#                 train_correct += 1
#
#         # print statistics every 10 batches
#         if (batch_index + 1) % 10 == 0:
#             print(f'Epoch {epoch + 1}/{epochs}, ' +
#                   f'Batch {batch_index + 1}/{len(train_dataloader)}, ' +
#                   f'Train Loss: {(train_loss / 1):.5f}, ' +
#                   f'Train Accuracy: {(train_correct / train_total):.5f}')
#
#             train_loss, train_correct, train_total = 0, 0, 0
#
# torch.save(net.state_dict(), "lstm.pkl")
# Test


