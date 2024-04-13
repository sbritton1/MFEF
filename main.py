import numpy as np
import pandas as pd
from random import sample
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch import nn
import cv2
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from src.white_balance import white_balance
from src.mfef import MFEF 


base_input_folder = 'data/input'
base_output_folder = 'data/output'
x_folder = base_input_folder + '/distorted/'
x_folder_clahe = base_output_folder + '/clahe/'
x_folder_wb = base_output_folder + '/wb/'
y_folder = base_input_folder + '/enhanced/'
model_folder = 'data/models/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # pre-processing
    apply_white_balance()
    apply_clahe()

    mfef = MFEF()
    mfef = mfef.to(device)

    writer = SummaryWriter()

    # Set the number of epochs to for training
    epochs = 100

    b_size = 1

    # Create loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(mfef.parameters(), lr=1e-4)

    dataset = ImageDataset(os.listdir(x_folder))

    train_set, test_set = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train_set, batch_size=b_size)
    test_loader = DataLoader(test_set, batch_size=b_size)

    print("Training:")
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        # Train on data
        train_loss, train_acc = train(train_loader,mfef,optimizer,criterion)

        # Test on data
        test_loss, test_acc = test(test_loader,mfef,criterion)

        # Write metrics to Tensorboard
        writer.add_scalars("Loss", {'Train': train_loss, 'Test':test_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc,'Test':test_acc} , epoch)
        path = os.path.join(model_folder, 'model_{}.pth'.format(epoch))
        torch.save(mfef.cpu().state_dict(), path) # saving model
        mfef.cuda() # moving model to GPU for further training

    print('Finished Training')
    writer.flush()
    writer.close()

def apply_white_balance():
    if not os.path.isdir(x_folder_wb):
        os.mkdir(x_folder_wb)

    print("Applying white balance:")
    for filename in tqdm(os.listdir(x_folder)):
        if os.path.exists(os.path.join(x_folder_wb, filename)):
            continue
        image = cv2.imread(os.path.join(x_folder, filename))
        final_img = white_balance(image)
        plt.imsave(os.path.join(x_folder_wb, filename),final_img)

def apply_clahe():
    if not os.path.isdir(x_folder_clahe):
        os.mkdir(x_folder_clahe)
    
    #parameters
    clip_limit = 40
    tile_grid_size=(8,8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    print("Applying CLAHE:")
    for filename in tqdm(os.listdir(x_folder)):
        if os.path.exists(os.path.join(x_folder_clahe, filename)):
            continue
        image = cv2.imread(os.path.join(x_folder, filename), cv2.IMREAD_COLOR)
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        image_lab[:,:,0] = clahe.apply(image_lab[:,:,0])
        final_img = cv2.cvtColor(image_lab, cv2.COLOR_Lab2RGB)
        plt.imsave(os.path.join(x_folder_clahe, filename),final_img)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_names):
        self.input_file_names = [os.path.join(x_folder, file_name) for file_name in file_names]
        self.label_file_names = [os.path.join(y_folder, file_name) for file_name in file_names]
        self.wb_file_names =    [os.path.join(x_folder_wb, file_name) for file_name in file_names]
        self.clahe_file_names = [os.path.join(x_folder_clahe, file_name) for file_name in file_names]

    def __len__(self):
        return len(self.input_file_names)

    def __getitem__(self, idx):
        image_orig = Image.open(self.input_file_names[idx])
        image_wb = Image.open(self.wb_file_names[idx])
        image_clahe = Image.open(self.clahe_file_names[idx])

        inputs = [image_orig, image_wb, image_clahe]

        convert_tensor = transforms.ToTensor()

        inputs = [convert_tensor(x) for x in inputs]

        # remove alpha channel in white balance
        for i in range(len(inputs)):
          x = inputs[i]
          if x.shape[0] == 4:
            inputs[i] = x[:3]

        label = convert_tensor(Image.open(self.label_file_names[idx]))
        if label.shape[0] == 4:
            label = label[:3]

        return inputs, label

def train(train_loader, net, optimizer, criterion):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: The model.
        optimizer: Optimizer.
        criterion: Loss function.
    """

    avg_loss = 0
    correct = 0
    total = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.squeeze().to(device, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs[0].squeeze().to(device, dtype=torch.float), inputs[1].squeeze().to(device, dtype=torch.float), inputs[2].squeeze().to(device, dtype=torch.float))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss
        predicted = outputs.data
        total += 1
        correct += (predicted == labels).sum().item()

    return avg_loss/len(train_loader), 100 * correct / total

def test(test_loader, net, criterion):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        net: Neural network model.
        criterion: Loss function.
    """

    avg_loss = 0
    correct = 0
    total = 0

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        for data in test_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.squeeze().to(device, dtype=torch.float)

            # forward pass
            outputs = net(inputs[0].squeeze().to(device, dtype=torch.float), inputs[1].squeeze().to(device, dtype=torch.float), inputs[2].squeeze().to(device, dtype=torch.float))
            loss = criterion(outputs, labels)

            # keep track of loss and accuracy
            avg_loss += loss
            predicted = outputs.data
            total += 1
            correct += (predicted == labels).sum().item()

    return avg_loss/len(test_loader), 100 * correct / total

if __name__ == "__main__":
    main()