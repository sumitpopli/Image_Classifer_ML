import torch
import PIL
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbrn
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
import json

# Device agnostic code, automatically uses CUDA if it's enabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def disable_GPU():
    device = torch.device("cpu")

def validate_cmdlineargs(cmdlineparams):
    
    desc = "All good"
    
    if(len(cmdlineparams.datadir) <= 0):
        desc = "No valid data dir passed."
        return False, desc
    
    if(cmdlineparams.lr>=1.0):
        desc = "Learning rate too high"
        return False, desc
    
    if(cmdlineparams.epochs <=0):
        desc = "epochs must be above 0"
        return False, desc
    
    if(cmdlineparams.dropout>=1.0):
        desc = "Dropout value should be less than 1.0"
        return False, desc
    
    if((cmdlineparams.hiddenunits < 102) or (cmdlineparams.hiddenunits > 4096) ):
        if((cmdlineparams.hiddenunits == 0)):
            desc= "All good"
        else:
            desc = "hidden units value should be between 102 and 4096"
            return False, desc
    return True, desc
    
def load_datatransforms(train_dir, valid_dir, test_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]

                                        )

    test_transforms = transforms.Compose(   [transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.Resize(255),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ]
                                        )
    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = data_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=50, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=50, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=50)
    
    return train_dataloaders, valid_dataloaders, test_dataloaders, train_datasets

def load_category_names():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print(cat_to_name)
    return cat_to_name
    
def create_Imageclassfier_NN(dropout, useVGG16, hiddenunits ):
    
    if(useVGG16 == False):
        model = models.vgg19(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    if(hiddenunits == 0):
        classifier = nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(25088,4096)),
                ('reLU1',nn.ReLU()),
                ('dropout1', nn.Dropout(p=dropout)),
                ('fc4', nn.Linear(4096, 102)),
                ('output', nn.LogSoftmax(dim=1))
            ]))
    else:    
        classifier = nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(25088,4096)),
                ('reLU1',nn.ReLU()),
                ('dropout1', nn.Dropout(p=dropout)),
                ('fc2',nn.Linear(4096, hiddenunits)),
                ('reLU2',nn.ReLU()),
                ('dropout2', nn.Dropout(p=dropout)),
                ('fc4', nn.Linear(hiddenunits, 102)),
                ('output', nn.LogSoftmax(dim=1))
            ]))
    
    model.classifier = classifier
    return model

# Implement a function for the validation pass
def validate_model(model, validloader, criterion):
    test_loss = 0
    accuracy = 0
    
    for v_images, v_labels in validloader:
        
        v_images, v_labels = v_images.to(device), v_labels.to(device)
        
        v_logps = model(v_images)
        v_loss = criterion(v_logps, v_labels)
        test_loss += v_loss.item()
        
        ps = torch.exp(v_logps)
        top_ps, top_class = ps.topk(1, dim=1)
        equality = top_class == v_labels.view(*top_class.shape)
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def test_network(model, test_dataloaders, criterion):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    test_equality = False
    with torch.no_grad():
        for t_images, t_labels in test_dataloaders:
                t_images, t_labels = t_images.to(device), t_labels.to(device)
                t_logps = model(t_images)
                t_loss = criterion(t_logps, t_labels)
                test_loss += t_loss.item()

                t_ps = torch.exp(t_logps)
                test_top_ps, test_top_class = t_ps.topk(1, dim=1)
                test_equality = test_top_class == t_labels.view(*test_top_class.shape)
                test_accuracy += torch.mean(test_equality.type(torch.FloatTensor)).item()
                
    print(f"Test loss: {test_loss/len(test_dataloaders):.3f}"
            f"Test accuracy: {test_accuracy/len(test_dataloaders):.3f}")

def train_network(lrate, e_count, train_dataloaders, valid_dataloaders,model,useSGDoptim):
    criterion = nn.NLLLoss()
    
    if(useSGDoptim == False):
        optimizer = optim.Adam(model.classifier.parameters(), lr = lrate)
    else: 
        optimizer = optim.SGD(model.classifier.parameters(), lr = lrate)
        
    model.to(device)
    # Define deep learning method
    epochs = e_count
    print_every = 30 # Prints every 30 images out of batch of 50 images
    steps = 0
    # Train the classifier layers using backpropogation using the pre-trained network to get features

    print("Training network start .....\n")

    for epoch in range(epochs):
        running_loss = 0
        model.train() 

        for images, labels in train_dataloaders:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validate_model(model, valid_dataloaders, criterion)

                print("Epoch: {}/{} | ".format(epoch+1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss/print_every),
                      "Validation Loss: {:.4f} | ".format(valid_loss/len(valid_dataloaders)),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(valid_dataloaders)))

                running_loss = 0
                model.train()

    print("\nTraining the network is now complete!!")
    return model, optimizer, criterion

def save_model_optimizer(save_location, train_datasets, model,optimizer, epochs, lr, isvgg16):
    
    save_path = save_location + 'checkpoint.pth'
    model.class_to_idx = train_datasets.class_to_idx
    model_name = ''
    if (isvgg16 == False):
        model_name = 'vgg19'
    else:
        model_name = 'vgg16'

    checkpoint ={'modelclassifier':model.classifier,
                 'state_dict':model.state_dict(),
                 'opt_dict': optimizer.state_dict(),
                 'epochs': epochs,
                 'classtoidx':train_datasets.class_to_idx,
                 'tvisionarch':model_name,
                 'pretrained': True,
                 'learnrate':lr
                }
    torch.save(checkpoint, save_path)
    print("saved the image classifier model at {}".format(save_path))
    return save_path


