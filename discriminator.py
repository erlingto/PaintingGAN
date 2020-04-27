import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from imutils import paths
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import cv2
from PIL import Image
import glob
import random

def calculate_conv_output(W, K, P, S):
    return int((W-K+2*P)/S)+1
    
def calculate_flat_input(dim_1, dim_2, dim_3):
    return int(dim_1*dim_2*dim_3)

class DiscriminatorNet(nn.Module):

    def __init__(self, img_rows, img_cols, channels):
        super(DiscriminatorNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 1, kernel_size = 3, stride= 1, padding= 1)
        self.pool1 = nn.MaxPool2d(2,2, padding= 0)
        self.conv_output_H = calculate_conv_output(img_rows, 2, 0, 2)
        self.conv_output_W = calculate_conv_output(img_cols, 2, 0, 2)

        self.fc1 = torch.nn.Linear(calculate_flat_input(1, self.conv_output_H, self.conv_output_W),  128)
        self.fc2 = torch.nn.Linear(128, 1)


    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x=self.pool1(x)
        x = x.view(-1, calculate_flat_input(1, self.conv_output_H, self.conv_output_W))

        x= F.relu(self.fc1(x))
        x= self.fc2(x)
        return x

class Discriminator:
    def __init__(self, learning_rate):
        self.model = DiscriminatorNet(64, 128, 3)
        self.images = ["drawing", "iconography", "painting", "sculpture"]
        drawing =  glob.glob("/Users/Erling/Documents/PaintingGAN/drawings/*")
        sculpture =  glob.glob("/Users/Erling/Documents/PaintingGAN/sculpture/*")
        painting =  glob.glob("/Users/Erling/Documents/PaintingGAN/painting/*")
        iconography =  glob.glob("/Users/Erling/Documents/PaintingGAN/iconography/*")

        self.optimizer = optim.Adam(self.model.parameters() ,lr = learning_rate)
        self.criterion = nn.MSELoss()

        self.list_of_paths = {"drawing": drawing, "sculpture": sculpture, "painting": painting, "iconography": iconography}
        self.batch_images = {}
        self.batch_labels = {}
        self.batch_path = {}

    def reset_batches(self):
        self.big_batch_images = {}
        self.big_batch_labels = {}
        self.batch_path = {}

    def reset_epoch(self): 
        drawing =  glob.glob("/Users/Erling/Documents/PaintingGAN/drawings/*")
        sculpture =  glob.glob("/Users/Erling/Documents/PaintingGAN/sculpture/*")
        painting =  glob.glob("/Users/Erling/Documents/PaintingGAN/painting/*")
        iconography =  glob.glob("/Users/Erling/Documents/PaintingGAN/iconography/*")

        self.list_of_paths = {"drawing": drawing, "sculpture": sculpture, "painting": painting, "iconography": iconography}

    def predict(self, image):
        return self.model(image)

    def load_images(self):
        size= 128, 64
        counter = 0
        for i in range(100):
            group = random.choice(self.images)

            while not self.list_of_paths[group]:
                self.images.remove(group)
                group = random.choice(self.images)

            imagePath = np.random.choice(self.list_of_paths[group])
            # load the image, pre-process it, and store it in the data list
            im = Image.open(imagePath)
            im.thumbnail(size, Image.ANTIALIAS)
            im = np.array(im)
            im = cv2.resize(im, (64, 128)) 
            im = torch.from_numpy(im)
            im = im.transpose(0,-1)
            im = im[None, :, :]

            while im.size() != torch.Size([1, 3, 64, 128]):
                imagePath = np.random.choice(self.list_of_paths[group])
                # load the image, pre-process it, and store it in the data list
                im = Image.open(imagePath)
                im.thumbnail(size, Image.ANTIALIAS)
                im = np.array(im)
                im = cv2.resize(im, (64, 128)) 
                im = torch.from_numpy(im)
                im = im.transpose(0,-1)
                im = im[None, :, :]

            if group == "painting":
                label = 1
            else:
                label = 0

            self.batch_images.update({str(counter): im})
            self.batch_labels.update({str(counter): label})
            self.batch_path.update({str(counter): imagePath})
            counter+= 1
            self.list_of_paths[group].remove(imagePath)
    
    def train(self, number_of_batches, number_of_epochs):
        loss_list = []
        acc_list = []
        batch_size = 100
        for epoch in range(number_of_epochs):
            for b in range(number_of_batches):
                correct = 0
                number_of_paintings = 0
                for i in range(batch_size):
                    # Run the forward pass
                    if self.batch_images[str(i)].size() != torch.Size([1, 3, 64, 128]):
                        print(self.batch_path[str(i)])
                    output = self.model(self.batch_images[str(i)])
                    
                    label = self.batch_labels[str(i)]
                    if label == 1:
                        number_of_paintings += 1
                    label = torch.Tensor([label])
                    label = label[None, :]
                    loss = self.criterion(output, label)
                    loss_list.append(loss.item())

                    # Backprop and perform Adam optimisation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Track the accuracy
                    total = 100
                    predicted = torch.round(output.data[0])
                    if predicted == label:
                        correct+= 1
                    acc_list.append(correct / total)

                    if (i + 1) % 100 == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                                .format(epoch + 1, number_of_epochs, i , i+1, loss.item(),
                                        (correct / total) * 100))
                        print(number_of_paintings)
                self.reset_batches()
                self.load_images()
            self.reset_epoch()


DClassifier = Discriminator(0.000146)
DClassifier.load_images()
DClassifier.train(20, 40)



#DClassifier.load_images()




