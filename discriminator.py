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

        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 32, kernel_size = 11, stride= 3, padding= 2)
        self.pool1 = nn.MaxPool2d(3,2, padding= 1)
        self.conv2 = nn.Conv2d(in_channels = 32 ,out_channels = 96, kernel_size = 3, stride= 2, padding= 1)
        self.pool2 = nn.MaxPool2d(2,2, padding= 1)
        self.conv3 = nn.Conv2d(in_channels = 96 ,out_channels = 192, kernel_size = 3, stride= 2, padding= 1)
        self.pool3 = nn.MaxPool2d(2,2, padding= 1)
        """ CONV DIMENSIONS CALCULATIONS """
        self.conv_output_H = calculate_conv_output(img_rows, 11, 2, 3)
        self.conv_output_W = calculate_conv_output(img_cols, 11, 2, 3)
        """ POOLING DIMENSIONS CALCULATIONS """
        self.conv_output_H = calculate_conv_output(self.conv_output_H , 3, 1, 2)
        self.conv_output_W = calculate_conv_output(self.conv_output_W , 3, 1, 2)
        
        for _ in range(2):
            self.conv_output_H = calculate_conv_output(self.conv_output_H , 3, 1, 2)
            self.conv_output_W = calculate_conv_output(self.conv_output_W , 3, 1, 2)
            self.conv_output_H = calculate_conv_output(self.conv_output_H , 2, 1, 2)
            self.conv_output_W = calculate_conv_output(self.conv_output_W , 2, 1, 2)
        
        self.classifier = nn.Sequential(
            torch.nn.Linear(calculate_flat_input(1, self.conv_output_H, self.conv_output_W)*192,  512), nn.ReLU(True),
            nn.Dropout(), nn.Linear(512, 4),)

        

    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x=self.pool1(x)
        x = F.relu(self.conv2(x.float()))
        x=self.pool2(x)
        x = F.relu(self.conv3(x.float()))
        x=self.pool3(x)
        x = x.view(-1, calculate_flat_input(1, self.conv_output_H, self.conv_output_W)*192)
        x= self.classifier(x)
        return x

class Discriminator:
    def __init__(self, learning_rate):
        self.model = DiscriminatorNet(512, 512, 3)
        #self.images = ["drawing", "iconography", "painting", "sculpture"]
        self.images = ["alfred", "leonardo", "pablo", "rembrandt"]
        alfred =  glob.glob("/Users/Erling/Documents/PaintingGAN/Painters/Alfred_Sisley/*")
        leonardo =  glob.glob("/Users/Erling/Documents/PaintingGAN/Painters/Leonardo_da_Vinci/*")
        pablo =  glob.glob("/Users/Erling/Documents/PaintingGAN/Painters/Pablo_Picasso/*")
        rembrandt =  glob.glob("/Users/Erling/Documents/PaintingGAN/Painters/Rembrandt/*")

        self.optimizer = optim.Adam(self.model.parameters() ,lr = learning_rate)
        self.criterion = nn.MSELoss()

        self.list_of_paths = {"alfred": alfred, "leonardo": leonardo, "pablo": pablo, "rembrandt": rembrandt}
        self.batch_images = {}
        self.batch_labels = {}
        self.batch_path = {}

    def reset_batches(self):
        self.big_batch_images = {}
        self.big_batch_labels = {}
        self.batch_path = {}
        self.images = ["alfred", "leonardo", "leonardo", "pablo", "rembrandt"]

    def reset_epoch(self): 
        alfred =  glob.glob("/Users/Erling/Documents/PaintingGAN/Painters/Alfred_Sisley/*")
        leonardo =  glob.glob("/Users/Erling/Documents/PaintingGAN/Painters/Leonardo_da_Vinci/*")
        pablo =  glob.glob("/Users/Erling/Documents/PaintingGAN/Painters/Pablo_Picasso/*")
        rembrandt =  glob.glob("/Users/Erling/Documents/PaintingGAN/Painters/Rembrandt/*")

        self.list_of_paths = {"alfred": alfred, "leonardo": leonardo, "pablo": pablo, "rembrandt": rembrandt}
        self.images = ["alfred", "leonardo", "leonardo", "pablo", "rembrandt"]
    def predict(self, image):
        return self.model(image)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.name = path

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def predict_test(self, imagePath):
        size= 512, 512
        im = Image.open(imagePath)
        im.thumbnail(size, Image.ANTIALIAS)
        im = np.array(im)
        im = cv2.resize(im, (512, 512)) 
        im = torch.from_numpy(im)
        im = im.transpose(0,-1)
        im = im[None, :, :]

        output = self.model(im)
        predicted = (output.data[0])
        if np.argmax(predicted) == 0:
            print("Bildet er en alfred")
        elif np.argmax(predicted) == 1:
            print("Bildet er en leo")
        elif np.argmax(predicted) == 2:
            print("Bildet er en pablo")
        elif np.argmax(predicted) == 3:
            print("Bildet er en Rembrand")
        print(predicted)
        return predicted


    def load_images(self):
        size= 512, 512
        counter = 0
        for i in range(32):
            group = random.choice(self.images)
            while not self.list_of_paths[group]:
                self.images.remove(group)
                group = random.choice(self.images)
                print("sletta")
            imagePath = np.random.choice(self.list_of_paths[group])
            # load the image, pre-process it, and store it in the data list
            im = Image.open(imagePath)
            im.thumbnail(size, Image.ANTIALIAS)
            im = np.array(im)
            im = cv2.resize(im, (512, 512)) 
            im = torch.from_numpy(im)
            im = im.transpose(0,-1)
            im = im[None, :, :]

            while im.size() != torch.Size([1, 3, 512, 512]):
                imagePath = np.random.choice(self.list_of_paths[group])
                # load the image, pre-process it, and store it in the data list
                im = Image.open(imagePath)
                im.thumbnail(size, Image.ANTIALIAS)
                im = np.array(im)
                im = cv2.resize(im, (512, 512)) 
                im = torch.from_numpy(im)
                im = im.transpose(0,-1)
                im = im[None, :, :]

            label = np.zeros(4)

            if group == "alfred":
                label[0] = 1
            elif group == "leonardo":
                label[1] = 1
            elif group == "pablo":
                label[2] = 1
            elif group == "rembrandt":
                label[3] = 1


            self.batch_images.update({str(counter): im})
            self.batch_labels.update({str(counter): label})
            self.batch_path.update({str(counter): imagePath})
            counter+= 1
    
    def train(self, number_of_batches, number_of_epochs):
        loss_list = []
        acc_list = []
        batch_size = 32
        for epoch in range(number_of_epochs):
            for b in range(number_of_batches):
                correct = 0
                number_of_leo_paintings = 0
                for i in range(batch_size):
                    # Run the forward pass
                    if self.batch_images[str(i)].size() != torch.Size([1, 3, 512, 512]):
                        print(self.batch_path[str(i)])
                    output = self.model(self.batch_images[str(i)])
                    
                    label = self.batch_labels[str(i)]
                    if label[1] == 1:
                        number_of_leo_paintings += 1
                    label = torch.Tensor([label])
                    loss = self.criterion(output, label)
                    loss_list.append(loss.item())

                    # Backprop and perform Adam optimisation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                

                    # Track the accuracy
                    total = 32
                    predicted = torch.round(output.data[0])
                    if np.argmax(predicted) == np.argmax(label):
                        correct+= 1
                    acc_list.append(correct / total)

                if (i + 1) % 32 == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                                .format(epoch + 1, number_of_epochs, b , number_of_batches, loss.item(),
                                        (correct / total) * 100))
                        print(number_of_leo_paintings)
                self.reset_batches()
                self.load_images()
            self.reset_epoch()

        self.save_weights("discriminator1")



def Mona_Lisa_Testen(Discriminator):
    image = "Mona_LisaTesten.jpg"
    print(Discriminator.predict_test(image))
    

def evaluation(Discriminator):

    alfred =  glob.glob("/Users/Erling/Documents/PaintingGAN/test/Alfred_Sisley/*")
    leonardo =  glob.glob("/Users/Erling/Documents/PaintingGAN/test/Leonardo_da_Vinci/*")
    pablo =  glob.glob("/Users/Erling/Documents/PaintingGAN/test/Pablo_Picasso/*")
    rembrandt =  glob.glob("/Users/Erling/Documents/PaintingGAN/test/Rembrandt/*")
    groups = ["alfred", "leonardo", "pablo", "rembrandt"]
    list_of_paths = {"alfred": alfred, "leonardo": leonardo, "pablo": pablo, "rembrandt": rembrandt}
    size= 512, 512
    counter = 0
    batch_images = {}
    batch_labels = {}
    error_results = {}
    for group in groups:
        for imagePath in list_of_paths[group]:
            # load the image, pre-process it, and store it in the data list
            im = Image.open(imagePath)
            im.thumbnail(size, Image.ANTIALIAS)
            im = np.array(im)
            im = cv2.resize(im, (512, 512)) 
            im = torch.from_numpy(im)
            im = im.transpose(0,-1)
            im = im[None, :, :]

            while im.size() != torch.Size([1, 3, 512, 512]):
                imagePath = np.random.choice(list_of_paths[group])
                # load the image, pre-process it, and store it in the data list
                im = Image.open(imagePath)
                im.thumbnail(size, Image.ANTIALIAS)
                im = np.array(im)
                im = cv2.resize(im, (512, 512)) 
                im = torch.from_numpy(im)
                im = im.transpose(0,-1)
                im = im[None, :, :]

            label = np.zeros(4)

            if group == "alfred":
                label[0] = 1
            elif group == "leonardo":
                label[1] = 1
            elif group == "pablo":
                label[2] = 1
            elif group == "rembrandt":
                label[3] = 1



            batch_images.update({str(counter): im})
            batch_labels.update({str(counter): label})
            
            counter+= 1
    
    print("Cross validation:")
    correct = 0
    total = 0
    alfred_error = 0
    leo_error = 0
    pablo_error = 0
    rembrandt_error = 0
    for i in range(16*4):
        output = DClassifier.model(batch_images[str(i)]).detach()
        predicted = np.argmax(output)
        label = batch_labels[str(i)]
        label = np.argmax(torch.Tensor([label]))
        

        if predicted == label:
            correct += 1
        else:
            if label == 0:
                alfred_error +=1
                error_results.update({("alfred" + str(alfred_error)) : output } )
            elif label == 1:
                leo_error+=1 
                error_results.update({("Da Vinci" + str(leo_error)) : output } )
            elif label == 2:
                pablo_error +=1
                error_results.update({("Picasso" + str(pablo_error)) : output } )
            elif label == 3:
                rembrandt_error += 1
                error_results.update({("Rembrandt" + str(rembrandt_error)) : output } )
        
        total += 1
    print("Cross validation:",correct/(total))
    print("The agent managed", correct, "out of a total of:", total)        
    print("Errors in Alfred images:", alfred_error, "out of 16")
    print("Errors in Da Vinci images:", leo_error, "out of 16")
    print("Errors in Picasso images:", pablo_error, "out of 16")
    print("Errors in Rembrandt images:", rembrandt_error, "out of 16")


    print ("-----------------------ERRORS-----------------------")
    for error in error_results:
        print(error ,":", error_results[error])

DClassifier = Discriminator(0.000146)
evaluation(DClassifier)
DClassifier.load_images()
DClassifier.train(16, 55)
DClassifier.load_weights("discriminator1")
evaluation(DClassifier)

#DClassifier.load_images()




