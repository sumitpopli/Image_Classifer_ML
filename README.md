# Image_Classifer_ML
Trains an image classifier to recognize different species of flowers.

## Description ##
This project implements a Neural Network image classifier. The aim of the classifier is to identify a flower picture correctly. 

## The project implements: ##
* **Training data augmentation :** Torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping

* **Data normalization:** The training, validation, and testing data is appropriately cropped and normalized

* **Data batching:** The data for each set is loaded with torchvision's DataLoader

* **Data loading:** The data for each set (train, validation, test) is loaded with torchvision's ImageFolder

* **Pretrained Network:** A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen

* **Feedforward Classifier:** A new feedforward network is defined for use as a classifier using the features as input

* **Training the network:** The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static

* **Testing Accuracy:** The network's accuracy is measured on the test data

* **Validation Loss and Accuracy:** During training, the validation loss and accuracy are displayed

* **Loading checkpoints:** There is a function that successfully loads a checkpoint and rebuilds the model

* **Saving the model:** The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary

* **Image Processing:** The process_image function successfully converts a PIL image into an object that can be used as input to a trained model

* **Class Prediction:** The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image

* **Sanity Checking with matplotlib:** A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names

## Installation ##
There are multiple ways to run this project:
1. Jupyter notebook
2. Python files

## The following python packages reqd to run this project: ##
1. Sklearn
2. Pandas, Numpy
3. Matplotlib
4. Torch

## Usage: ##

After the environment set up,
- Download flowers.zip. and extract zip file in the same folders as the project. 

**using Jupyter file:**
	- Load up Image Classifier project.ipynb
	- make sure the paths are correct before executing the whole project.
  
**using Python files:**

	The project is divided in to two parts:
	-train - 
		contains two files: train.py, train_functions.py
		training data is in /train dir
		the code trains the image classifier and stores the classifier as checkpoint.pth
	-predict
		contains two files: predict.py, predict_functions.py
		test data is in /test dir
		code loads the stored classifier object and classifies images. 
	Both file have command line functions: Type the following in the terminal windows to get commandline syntax
		python train.py --help
		python predict.py --help 



