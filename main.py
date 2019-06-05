import unet_model
from data_helper import adjustDataTest, dataGenerator,load_data_Kfold, get_items

BATCH_SIZE = 2

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

#Create folds
x_train,x_validation,y_train,y_validation = load_data_Kfold(im_path,label_path,k)

#Training 
model = unet_model.unet()

for fold_number in range(k):
    x_training = get_items(x_train[fold_number])
    y_training = get_items(y_train[fold_number])
    x_valid = get_items(x_validation[fold_number])
    y_valid = get_items(y_validation[fold_number])
    gen_test = adjustDataTest(x_valid,y_valid)
    print(f'Training fold {fold_number}')
    generator = dataGenerator(BATCH_SIZE, x_training,y_training,data_gen_args,seed = 1) 
   # model.fit(x_training,y_training,steps_per_epoch=int(len(x_training)/BATCH_SIZE),epochs=1,verbose=1,validation_steps = int(len(x_valid)/BATCH_SIZE),validation_data = (x_valid,y_valid))
    model.fit_generator(generator, steps_per_epoch = len(x_training) // BATCH_SIZE,epochs=1,verbose=1,validation_steps = len(x_valid)// BATCH_SIZE, validation_data=gen_test)



#results = model.predict_generator(testGene,10,verbose=1)
#data_helper.saveResult("data/test",results)
