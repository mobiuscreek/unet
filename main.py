import unet_model
import data_helper
        
BATCH_SIZE = 2
        
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = data_helper.dataGenerator(BATCH_SIZE,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet_model.unet()
#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit_generator(myGene,steps_per_epoch=1,epochs=3)

figure = data_helper.plot_metrics(history)

testGene = data_helper.test_file_reader("data/membrane/test")

#validation set
loss, acc = model.evaluate_generator(myGene, steps=10)

results = model.predict_generator(testGene,10,verbose=1)
data_helper.saveResult("data/test",results)
