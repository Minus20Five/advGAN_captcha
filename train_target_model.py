import os
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from models import MNIST_target_net
from PIL import Image
import numpy as np
from generate_captchas import LABEL_SEQ_VALUE, LABEL_SEQ_LENGTH

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class CaptchaDataset(Dataset):
    """Loads Captcha from directory"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [name for name in os.listdir(root_dir) if ".png" in name]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("L")
        label = os.path.splitext(self.image_files[idx])[0]
        label_one_hot = np.zeros(len(LABEL_SEQ_VALUE) * LABEL_SEQ_LENGTH)

        for i, char in enumerate(label):
            label_one_hot[i * LABEL_SEQ_LENGTH + LABEL_SEQ_VALUE.find(char)] = 1

        image = self.transform(image)
        label_one_hot = torch.from_numpy(label_one_hot).float()
        return image, label_one_hot


if __name__ == "__main__":
    use_cuda = True
    image_nc = 1
    batch_size = 256

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    captcha_dataset = CaptchaDataset(os.path.join(FILE_DIR, "data", "train"))
    train_dataloader = DataLoader(captcha_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # training the target model
    target_model = MNIST_target_net(classes=len(LABEL_SEQ_VALUE), label_size=LABEL_SEQ_LENGTH).to(device)
    target_model.train()
    opt_model = torch.optim.Adam(target_model.parameters(), lr=0.001)
    epochs = 1
    for epoch in range(epochs):
        loss_epoch = 0
        if epoch == 20:
            opt_model = torch.optim.Adam(target_model.parameters(), lr=0.0001)
        for char_sec, data in enumerate(train_dataloader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            logits_model = target_model(train_imgs)
            loss_model = F.binary_cross_entropy_with_logits(logits_model, train_labels)
            loss_epoch += loss_model
            opt_model.zero_grad()
            loss_model.backward()
            opt_model.step()
        print('loss in epoch %d: %f' % (epoch, loss_epoch.item()))

    # save model
    targeted_model_file_name = './MNIST_target_model.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()

    # MNIST test dataset
    captcha_dataset_test = CaptchaDataset(os.path.join(FILE_DIR, "data", "test"))
    test_dataloader = DataLoader(captcha_dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred = target_model(test_img)
        pred = pred.detach().numpy()

        pred_one_hot = np.zeros((pred.shape[0], len(LABEL_SEQ_VALUE) * LABEL_SEQ_LENGTH))
        for pred_row in enumerate(pred):
            for char_sec in range(LABEL_SEQ_LENGTH):
                slice = pred_row[char_sec * len(LABEL_SEQ_VALUE): (char_sec + 1) * len(LABEL_SEQ_VALUE)]
                pred_one_hot[char_sec][char_sec * len(LABEL_SEQ_VALUE) + np.argmax(slice)] = 1

        # pred_lab = torch.argmax(target_model(test_img), 1)
        pred_one_hot = torch.from_numpy(pred_one_hot).float()
        num_correct += torch.sum(pred_one_hot == test_label, 0)

        print("shape {}".format(num_correct.shape))
        print('accuracy in testing set: %f\n' % (num_correct.item() / len(captcha_dataset_test)))
