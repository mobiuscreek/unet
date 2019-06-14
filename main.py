import unet_model
from data_helper import dataGenerator,load_data_Kfold, get_items, test_file_reader, saveResult



BATCH_SIZE = 2
        
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

im_path = 'data/membrane/train/im_test'
label_path = 'data/membrane/train/label_test'
im_test = 'data/membrane/train/im_test'

k = 5
seed = 1

#Create folds
folds, x_train, y_train = load_data_Kfold(im_path,label_path,k)

#Load model
model = unet_model.unet()

#CV and training
for fold_number, (train_idx,val_idx) in enumerate(folds)
    print(f'Training fold {fold_number}')
    x_training_filenames = x_train[train_idx]
    y_training_filenames = y_train[train_idx]
    x_valid_filenames = x_train[val_idx]
    y_valid_filenames = y_train[val_idx]
    x_training = get_items(x_training_filenames)
    y_training = get_items(y_training_filenames)
    x_valid = get_items(x_valid_filenames)
    y_valid = get_items(y_valid_filenames)
    generator = dataGenerator(BATCH_SIZE, x_training,y_training,data_gen_args,seed = 1) 
    model.fit_generator(generator,steps_per_epoch=len(x_training)/BATCH_SIZE,epochs=3,verbose=1,validation_data = (x_valid,y_valid))

#Read test data and evaluate
testGen = test_file_reader(im_test)

results = model.predict_generator(testGen,10,verbose=1)
saveResult("data/membrane/test",results)
