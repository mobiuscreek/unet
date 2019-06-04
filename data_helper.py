from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import skimage.transform as trans
import matplotlib.pyplot as plt
from glob import glob
import pdb

from skimage.color import rgb2gray

def adjustData(img,mask):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return (img,mask)



def dataGenerator(batch_size,X_train,Y_train,aug_dict,image_save_prefix  = "image", 
                    save_to_dir = None,seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow(
        X_train, Y_train,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    for (img,mask) in image_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)

from sklearn.model_selection import KFold
from skimage import io

def load_data_Kfold(path_X,path_Y,k):
    train_files = glob(os.path.join(path_X,'*.jpg'))
    train_labels = glob(os.path.join(path_Y,'*.jpg'))
    X_train_np = np.asarray(train_files)
    Y_train_np = np.asarray(train_labels)
    kf = KFold(n_splits=k,shuffle=True,random_state=1)
    X_valid = []
    X_train = []
    y_train = []
    y_valid = []
    X = np.zeros(shape=(1,1))
    X_validation = np.zeros(shape=(1,1))
    Y = np.zeros(shape=(1,1))
    Y_validation = np.zeros(shape=(1,1))
    for train_index, test_index in kf.split(X_train_np):
        X_train.append([np.sort(X_train_np[train_index])])
        X_valid.append([np.sort(X_train_np[test_index])])
        y_train.append([np.sort(Y_train_np[train_index])])
        y_valid.append([np.sort(Y_train_np[test_index])])
    return X_train, X_valid, y_train, y_valid    



def get_items(list_of_lists):
    image_list = [] 
    flat_list = [item for list_of_lists[0] in list_of_lists for item in list_of_lists[0]]
    for j in range(len(flat_list)):
        img = io.imread(flat_list[j],as_gray = True)
        img = trans.resize(img,(256,256))
        image_list.append(img)
        image_np = np.asarray(image_list)
        image_np = np.expand_dims(image_np,axis=3)
    return image_np   



'''
for train_index, test_index in kf.split(X_train_np):
        if First_iteration:
            X_valid.append(X_train_np[test_index])
            X_train.append(X_train_np[train_index])
            y_train.append(Y_train_np[train_index])
            y_valid.append(Y_train_np[test_index])
            X_valid = np.asarray(X_valid)
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)
            y_valid = np.asarray(y_valid)
            First_iteration = False
        else:
          pdb.set_trace()
          X =  np.vstack((X_train,X_train_np[train_index].reshape(1,X_train_np[train_index].shape[0])))
          X_validation =  np.vstack((X_valid,X_train_np[test_index].reshape(1,X_train_np[test_index].shape[0])))
          Y =  np.vstack((y_train,Y_train_np[train_index].reshape(1,Y_train_np[train_index].shape[0])))
          Y_validation =  np.vstack((y_valid,Y_train_np[test_index].reshape(1,Y_train_np[test_index].shape[0])))
      #X_train.append(X_train_np[train_index])
        # np.append(X_valid,X_train_np[test_index])
        # np.append(y_train,Y_train_np[train_index])
        # np.append(y_valid,Y_train_np[test_index])
        #y_train, y_valid = Y_train_np[train_index], Y_train_np[test_index]
    return X, X_validation, Y, Y_validation
'''
def test_file_reader(test_path,target_size = (256,256),as_gray = True):
    '''
        Reads path, resized and returns all images on specified folder
    '''
    extensions = glob(os.path.join(test_path,'*.jpg'))
    for filename in extensions:
        img = io.imread(filename,as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
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
    


if __name__ == "__main__":
    load_data_Kfold('data\\membrane\\train\\image','data\\membrane\\train\\label',4)
    
