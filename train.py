import argparse
import json
import train_functions as tf


parser = argparse.ArgumentParser(description='Image Classifier Udacity')

parser.add_argument('-datadir', action="store", dest="datadir",required=True, help="Base image path where the images are. Path can be relative")
parser.add_argument('-chkpointdir', action="store", dest="chkpointdir", default='', help="Path where you want the model to be stored. Path can be relative or empty string")
parser.add_argument('-lr', action="store", dest="lr", type=float,default=0.0001, help="Learning rate of the Neural network. The value cannot be equal to or more than 1")
parser.add_argument('-epochs', action="store", dest="epochs", type=int,default=5, help="Number of times you want the Model to learn from the training data. Value must be more than 0")
parser.add_argument('-useVGG16', action="store_true", dest="useVGG16",default=False, help="Default model is vgg19. You can switch to vgg16 but using this option.")
parser.add_argument('-dropout', action="store", dest="dropout",type=float,default=0.5, help="Dropout value for backpropogation. Value must be float and less than 1")
parser.add_argument('-useSGDoptim', action="store_true", dest="useSGDoptim",default=False, help="Default Optimizer is Adam. Use this option to replace the default with SGD")
parser.add_argument('-noGPU', action="store_true", dest="noGPU", help="GPU-CUDA is on if available by default. Use this option to use CPU only")
parser.add_argument('-hiddenunits', action="store", dest="hiddenunits",type=int, default=0, help="To add additional hidden units. The value must be between 4096 and 102 or 0")



print(parser.parse_args())
image_NN_params = parser.parse_args()

retval, desc = tf.validate_cmdlineargs(image_NN_params)

if(retval == False):
    print(desc)

if(retval == True):
    data_dir = image_NN_params.datadir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    if(image_NN_params.noGPU == True):
        tf.disable_GPU()
    
    train_dl, valid_dl, test_dl, train_ds = tf.load_datatransforms(train_dir, valid_dir, test_dir)
    cat_to_name = tf.load_category_names()
    ic_model = tf.create_Imageclassfier_NN(image_NN_params.dropout, image_NN_params.useVGG16, image_NN_params.hiddenunits)
    ic_model, ic_optimizer, ic_criterion = tf.train_network(image_NN_params.lr, image_NN_params.epochs, train_dl,valid_dl, ic_model, image_NN_params.useSGDoptim)
    tf.test_network(ic_model, test_dl, ic_criterion)
    savepath_model = tf.save_model_optimizer(image_NN_params.chkpointdir, train_ds, ic_model, ic_optimizer, image_NN_params.epochs, image_NN_params.lr, image_NN_params.useVGG16)

