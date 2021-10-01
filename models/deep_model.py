import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch

class Trainmodel():
    def __init__(self,pretrained =True):
        self.model = models.densenet161(pretrained)
        for self.param in self.model.parameters():
            self.param.requires_grad = False

    def classified(self, num_labels):
        self.classifier_input = self.model.classifier.in_features
        self.num_labels = num_labels
        classifier = nn.Sequential(nn.Linear(self.classifier_input, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, num_labels),
                                   nn.LogSoftmax(dim=1))

        self.model.classifier = classifier
        return  self.model

    def training(self,model, train_loader, validation_loader, device ='cpu'):
        self.model = model
        self.epochs = 15
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.classifier.parameters())
        for epoch in range(self.epochs):
            train_loss = 0
            val_loss = 0
            accuracy = 0

            # Training the model
            self.model.train()
            counter = 0
            for inputs, labels in self.train_loader:
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Clear optimizers
                optimizer.zero_grad()
                # Forward pass
                output = self.model.forward(inputs)
                # Loss
                loss = criterion(output, labels)
                # Calculate gradients (backpropogation)
                loss.backward()
                # Adjust parameters based on gradients
                optimizer.step()
                # Add the loss to the training set's rnning loss
                train_loss += loss.item() * inputs.size(0)

                # Print the progress of our training
                counter += 1
                print(counter, "/", len(self.train_loader))

            # Evaluating the model
            self.model.eval()
            counter = 0
            # Tell torch not to calculate gradients
            with torch.no_grad():
                for inputs, labels in self.validation_loader:
                    # Move to device
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Forward pass
                    output = self.model.forward(inputs)
                    # Calculate Loss
                    valloss = criterion(output, labels)
                    # Add loss to the validation set's running loss
                    val_loss += valloss.item() * inputs.size(0)

                    # Since our model outputs a LogSoftmax, find the real
                    # percentages by reversing the log function
                    output = torch.exp(output)
                    # Get the top class of the output
                    top_p, top_class = output.topk(1, dim=1)
                    # See how many of the classes were correct?
                    equals = top_class == labels.view(*top_class.shape)
                    # Calculate the mean (get the accuracy for this batch)
                    # and add it to the running accuracy for this epoch
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    # Print the progress of our evaluation
                    counter += 1
                    print(counter, "/", len(self.validation_loader))

            # Get the average loss for the entire epoch
            train_loss = train_loss / len(self.train_loader.dataset)
            valid_loss = val_loss / len(self.validation_loader.dataset)
            # Print out the information
            print('Accuracy: ', accuracy / len(self.validation_loader))
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
            torch.save(self.model, 'SavedModel/model.pth')
            return 'SavedModel is saved'
