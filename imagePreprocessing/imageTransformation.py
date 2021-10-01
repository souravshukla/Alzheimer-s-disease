import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch

path = 'Dataset/axial'
class DataTransform():
    def __init__(self, batch_size, shuffle = True):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle

    def transformation(self, size):
        self.transform = transforms.Compose([transforms.Resize(size),
								        transforms.ToTensor(),
								        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        self.data_set = datasets.ImageFolder(self.path, transform = self.transform)
        self.train_set, self.validation_set = torch.utils.data.random_split(self.data_set,[5000,857]) #5000,870
        train_loader = DataLoader(dataset=self.train_set, shuffle=self.shuffle, batch_size=self.batch_size)
        validation_loader = DataLoader(dataset=self.validation_set, shuffle=self.shuffle, batch_size=self.batch_size)

        return train_loader, validation_loader

