from itertools import groupby

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from solver.another_cnn.lstm import StackedLSTM, CaptchaDataset, criterion, transform
from solver.captcha_setting import TEST_DATASET_PATH
from utils.utils import training_device

BLANK_LABEL = 10
BATCH_SIZE = 64
device = training_device()

net = StackedLSTM()
net.load_state_dict(torch.load('lstm.pkl', map_location=device))
net.eval()
h = net.init_hidden(BATCH_SIZE)  # init hidden state

test_dataloader = DataLoader(CaptchaDataset(root_dir=TEST_DATASET_PATH, transform=transform), batch_size=64, shuffle=True)

test_loss = 0
test_correct = 0
test_total = len(test_dataloader.dataset)

with torch.no_grad():  # detach gradients so network runs faster

    # for each batch of testing examples
    for batch_index, (inputs, targets) in enumerate(test_dataloader):

        inputs, targets = inputs.to(device), targets.to(device)
        h = tuple([each.data for each in h])
        batch_size, channels, height, width = inputs.shape

        # breaks otherwise (something to do with expected input of hidden layers)
        if len(targets) < BATCH_SIZE:
            continue

        # reshape inputs: NxCxHxW -> WxNx(HxC)
        inputs = (inputs
                  .permute(3, 0, 2, 1)
                  .contiguous()
                  .view((width, batch_size, -1)))

        outputs, h = net(inputs, h)  # forward pass

        # record loss
        input_lengths = torch.IntTensor(batch_size).fill_(width)
        target_lengths = torch.IntTensor([len(t) for t in targets])
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        test_loss += loss.item()

        # compare prediction with ground truth
        prob, max_index = torch.max(outputs, dim=2)

        for i in range(batch_size):
            raw_pred = list(max_index[:, i].cpu().numpy())
            pred = [c for c, _ in groupby(raw_pred) if c != BLANK_LABEL]
            target = list(targets[i].cpu().numpy())
            if pred == target:
                test_correct += 1

print(f'Test Loss: {(test_loss / len(test_dataloader)):.5f}, ' +
      f'Test Accuracy: {(test_correct / test_total):.5f} ' +
      f'({test_correct}/{test_total})')

# See how it is
data_iterator = iter(test_dataloader)
inputs, targets = data_iterator.next()

i = 1

image = inputs[i,0,:,:]

print(f"Target: {''.join(map(str, targets[i].numpy()))}")
plt.imshow(image)

inputs = inputs.to(device)

batch_size, channels, height, width = inputs.shape
h = net.init_hidden(batch_size)

inputs = (inputs
          .permute(3, 0, 2, 1)
          .contiguous()
          .view((width, batch_size, -1)))

# get prediction
outputs, h = net(inputs, h)  # forward pass
prob, max_index = torch.max(outputs, dim=2)
raw_pred = list(max_index[:, i].cpu().numpy())

# print raw prediction with BLANK_LABEL replaced with "-"
print('Raw Prediction: ' + ''.join([str(c) if c != BLANK_LABEL else '-' for c in raw_pred]))

pred = [str(c) for c, _ in groupby(raw_pred) if c != BLANK_LABEL]
print(f"Prediction: {''.join(pred)}")