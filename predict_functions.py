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



def validate_cmdlineargs(cmdlineparams):
    
    desc = "All good"
    
    if(len(cmdlineparams.datadir) <= 0):
        desc = "No valid data dir passed."
        return False, desc


    if(len(cmdlineparams.testimgpath) <= 0):
        desc = "No valid image path was passed."
        return False, desc
    
    if(cmdlineparams.topK < 1):
        desc = "TopK argument should be greater than or equal to 1"
        return False, desc
    
    return True, desc

def load_category_names():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_checkpoint(object_filepath):
    model_props = torch.load(object_filepath)
    model_chkpnt = None
    optimizer_chkpnt = None
    if (model_props['tvisionarch'] == 'vgg19'):
        
        print(model_props['pretrained'])
        model_chkpnt = models.vgg19(pretrained=model_props['pretrained'])
    else:
        print(model_props['pretrained'])
        model_chkpnt = models.vgg16(pretrained=model_props['pretrained'])
        
    for param in model_chkpnt.parameters():
        param.requires_grad = False
        
    model_chkpnt.classifier = model_props['modelclassifier']
        
    print(model_props['classtoidx'])
    model_chkpnt.class_to_idx = model_props['classtoidx']
        
             
    #print(model_props['state_dict'])
    model_chkpnt.load_state_dict(model_props['state_dict'])
        
    print(model_props['learnrate'])
    optimizer_chkpnt= optim.Adam(model_chkpnt.classifier.parameters(),model_props['learnrate'])
        
    #print(model_props['opt_dict'])
    optimizer_chkpnt.load_state_dict(model_props['opt_dict'])
        
              
    return model_chkpnt,optimizer_chkpnt
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = PIL.Image.open(image)
    im.show()
    
    # resize the image while maintaining the aspect ratio... for that find image dim and then resize
    width, height = im.size
    n_width = 0
    n_height = 0
    
    if(width >=  height):
        n_width = int((width/height)* 256)
        n_height = 256
    else:
        n_height = int((height/width)*256)
        n_width = 256
        
    print("original image dims:{0},{1}".format(height, width))
    print("new image dims:{0},{1}".format(n_height, n_width))
        
    im.resize((n_width, n_height))
    print("thumbnail dim {}".format(im.size))
    
    #now crop out 224,224
    #find the center point
    cpoint_x = int(n_width/2)
    cpoint_y = int(n_height/2)
    
    #get left, right, top, bottom
    left = int(cpoint_x  - (224/2))
    right = int(cpoint_x + (224/2))
    top = int(cpoint_y -(224/2))
    bottom = int(cpoint_y + (224/2))
        
    im_final = im.crop((left, top, right, bottom))
    print("final image dimensions {}",format(im_final.size))
    
    #converting 255 vals to equivalent float vals
    np_image = np.array(im_final)/255
    
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
    
    #0,1,2 to 2,0,1
    np_image = np_image.transpose(2,0,1)
    
    return torch.Tensor(np_image)
    
def predict(image_path, model, topk, noGPU):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    if(noGPU == False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device =torch.device("cpu")
        
    model.to(device)
    model.eval()
    
    img = process_image(image_path).type(torch.FloatTensor)
    img = img.unsqueeze_(0)
    logps = model(img.to(device))
    
    ps = torch.exp(logps)
    top_ps, top_class = ps.topk(topk, dim=1)
    
    print("top probabilities {}".format(top_ps))
    print("top classes {}".format(top_class))
    
    #inverting the dictionary
    idx_to_class = {}
    for key, val in model.class_to_idx.items():
        idx_to_class[val] = key
    
    top_class_labels=[]
    
    #get the error can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    # following error instructions 
    # refer (https://stackoverflow.com/questions/53467215/convert-pytorch-tensor-to-numpy-array-using-cuda)
    #converting all torch tensors to regular lists.
    
    top_class_np = top_class.cpu().detach().numpy().tolist()[0] 
    print(top_class_np)
    top_ps_np = top_ps.cpu().detach().numpy().tolist()[0]
    
    for class_val in top_class_np:
        top_class_labels.append(idx_to_class[class_val])
        
    
    print("top class label values {}".format(top_class_labels))
    cat_to_name  = load_category_names()
    
    top_flowers_class = []
    for label_vals in top_class_labels:
        top_flowers_class.append(cat_to_name[label_vals]) 
    
    print(top_flowers_class)
    
    return top_ps_np, top_class_labels, top_flowers_class    