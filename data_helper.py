import os
import matplotlib.pyplot as plt
import numpy as np 
import skimage.transform as trans

from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from skimage import io



def dataGenerator(batch_size,im_data ,label_data, aug_dict,image_save_prefix  = "image", 
                    save_to_dir = None,seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow(
        im_data, y=None,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow(
        label_data,y=None,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
            mask[mask>0.5]=1
            mask[mask<=0.5]=0
            yield (img, mask)       




def load_data_Kfold(path_X,path_Y,k):
    train_files = glob(os.path.join(path_X,'*.jpg'))
    train_labels = glob(os.path.join(path_Y,'*.jpg'))
    X_train_np = np.asarray(train_files)
    Y_train_np = np.asarray(train_labels)
    folds = list(KFold(n_splits=k,shuffle=True,random_state=1).split(X_train_np))
    return folds, X_train_np, Y_train_np 



def get_items(array_of_filenames, target_dim = (256,256)):
    image_list = []
    for j in range(len(array_of_filenames)):
        img = io.imread(array_of_filenames[j],as_gray = True)
        img = trans.resize(img, target_dim, mode='constant')
        image_list.append(img)
        image_np = np.asarray(image_list)
        image_np = np.expand_dims(image_np,axis=3)
    return image_np   




def test_file_reader(test_path, as_gray = True, target_dim = (256,256)):
    '''
        Reads path, resized and returns all images on specified folder
    '''
    extensions = glob(os.path.join(test_path,'*.jpg'))
    for filename in extensions:
        img = io.imread(filename,as_gray = as_gray)
        img = trans.resize(img, target_dim, mode='constant')
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img



def saveResult(save_path,pred_im_array): 
    #saves images into specified directory
    for i,item in enumerate(pred_im_array):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path,f"{i}_predict.png"),img)
        
 
def plot_metrics(history_obj):
    fig = plt.figure()
    plt.plot(history_obj.history['loss'])
    plt.plot(history_obj.history['acc']) 
    plt.title('model performance')  
    plt.xlabel('epoch')  
    plt.legend(['loss', 'accuracy'], loc='upper left') 
    fig.savefig('model_performance.png', dpi=1000)   
    
