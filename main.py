import unet_model
import data_helper
import pdb
        
BATCH_SIZE = 16
        
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

im_path = 'data/membrane/train/image'
label_path = 'data/membrane/train/label'
k = 2
seed = 1

x_train,x_validation,y_train,y_validation = data_helper.load_data_Kfold(im_path,label_path,k)
#Training 
model = unet_model.unet()
print(x_train[0][0][0:2])
print(y_train[0][0][0:2])
print(x_validation[0][0][0:2])
print(y_validation[0][0][0:2])

for fold_number in range(k):
    x_training = data_helper.get_items(x_train[fold_number])
    y_training = data_helper.get_items(y_train[fold_number])
    x_valid = data_helper.get_items(x_validation[fold_number])
    y_valid = data_helper.get_items(y_validation[fold_number])
    print(f'Training fold {fold_number}')
    generator = data_helper.dataGenerator(BATCH_SIZE, x_training,y_training,data_gen_args,seed = 1) 
    model.fit_generator(generator,steps_per_epoch=len(x_training)/BATCH_SIZE,epochs=1,verbose=1,validation_data = (x_valid,y_valid))




#results = model.predict_generator(testGene,10,verbose=1)
#data_helper.saveResult("data/test",results)
