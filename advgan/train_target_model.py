import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from advgan.models import MNIST_target_net
from PIL import Image
from generate_captchas import LABEL_SEQ_VALUE, LABEL_SEQ_LENGTH

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class CaptchaDataset(Dataset):
    """Loads Captcha from directory"""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [name for name in os.listdir(root_dir) if ".png" in name]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("L")
        label_str = os.path.splitext(self.image_files[idx])[0]
        label = torch.zeros(LABEL_SEQ_LENGTH).long()

        for i, char in enumerate(label_str):
            label[i] = LABEL_SEQ_VALUE.find(char)

        image = self.transform(image)
        return image, label


if __name__ == "__main__":
    use_cuda = True
    image_nc = 1
    batch_size = 32

    # # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())

    if (use_cuda and torch.cuda.is_available()) :
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = "cuda:0"
    else:
        device = "cpu"

    captcha_dataset = CaptchaDataset(os.path.join(FILE_DIR, "data", "train"))
    train_dataloader = DataLoader(captcha_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # training the target model
    target_model = MNIST_target_net(classes=len(LABEL_SEQ_VALUE), label_size=LABEL_SEQ_LENGTH)
    target_model = target_model.to(device)
    target_model.train()
    opt_model = torch.optim.Adam(target_model.parameters(), lr=0.001)
    epochs = 20
    for epoch in range(epochs):
        loss_epoch = 0
        if epoch == 20:
            opt_model = torch.optim.Adam(target_model.parameters(), lr=0.0001)
        for char_sec, data in enumerate(train_dataloader, 0):
            train_imgs, train_labels = data
            train_imgs = train_imgs.to(device)
            train_labels =  train_labels.to(device)
            logits_list = target_model(train_imgs)
            loss_model = 0
            for char_idx, logit in enumerate(logits_list):
                char_label = train_labels.index_select(dim=1, index=torch.tensor(char_idx)).squeeze()
                loss_model += F.cross_entropy(logit, char_label)
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
        test_img, test_labels = data
        test_img, test_labels = test_img.to(device), test_labels.to(device)
        logits_list = target_model(test_img)
        label_list = []
        for logit in logits_list:
            label_list.append(logit.argmax(dim=1))
        pred_labels = torch.stack(label_list, dim=1)
        num_correct += (pred_labels == test_labels).all(dim=1).sum(dim=0)

    print('accuracy in testing set: %f\n' % (num_correct.item() / len(captcha_dataset_test)))
