
import argparse
import json
import predict_functions as pf


parser = argparse.ArgumentParser(description='Image Classifier Udacity')

parser.add_argument('-datadir', action="store", dest="datadir",required=True)
parser.add_argument('-chkpointdir', action="store", dest="chkpointdir", default='')
parser.add_argument('-topK', action="store", dest="topK", type=int,default=5)
parser.add_argument('-testimgpath', action="store", dest="testimgpath",required = True)
parser.add_argument('-noGPU', action="store_true", dest="noGPU")





print(parser.parse_args())
pred_Image_params = parser.parse_args()

retval, desc = pf.validate_cmdlineargs(pred_Image_params)

if(retval == False):
    print(desc)
    



if(retval == True):
    
    data_dir = pred_Image_params.datadir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    chkpoint_path = pred_Image_params.chkpointdir + 'checkpoint.pth'
    modelv2, optimv2 = pf.load_checkpoint(chkpoint_path)
    pf.predict(pred_Image_params.testimgpath, modelv2, pred_Image_params.topK, pred_Image_params.noGPU)

'''
# command line arguments
python predict.py -datadir 'flowers' -topK 5 -testimgpath "flowers/test/10/image_07090.jpg"
python predict.py -datadir 'flowers' -topK 5 -testimgpath "flowers/test/12/image_04014.jpg"
python predict.py -datadir 'flowers' -topK 5 -testimgpath "flowers/test/11/image_03151.jpg"
'''    
    