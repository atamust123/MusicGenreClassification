#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras import layers
from tensorflow import keras
from keras.models import Sequential,Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os
from matplotlib import pyplot as plt
from natsort import natsorted
from tensorflow import keras
import numpy as np
import pandas as pd
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dropout,Input,Flatten, Dense, Activation, BatchNormalization
get_ipython().system('pip install natsort==3.3.0 #sort the files in the directory ')


# In[ ]:


def GenreModel(input_shape = (216,220,3),classes=10):
  
    X_input = Input(input_shape)

    X = Conv2D(8,kernel_size=(3,3),strides=(1,1))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(16,kernel_size=(3,3),strides = (1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Dropout(rate=0.25)(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(32,kernel_size=(3,3),strides = (1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(64,kernel_size=(3,3),strides=(1,1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(128,kernel_size=(3,3),strides=(1,1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Flatten()(X)
    X=Dense(256, activation='relu')(X)
    X = Dropout(rate=0.25)(X)
    X=Dense(128, activation='relu')(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)
    model = Model(inputs=X_input,outputs=X,name='GenreModel')

    return model


# In[ ]:


#model definition
model=GenreModel()
optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[ ]:


#dataset created according to kaggle input path

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    "../input/cropped-spects/train",
    shuffle=True,    target_size=(216,220),
    class_mode='categorical',
    batch_size=32
)
train_generator
validation_generator = datagen.flow_from_directory(
    "../input/cropped-spects/validation",
    shuffle=True,    target_size=(216,220),
    class_mode='categorical',
    batch_size=32
)

test_generator = datagen.flow_from_directory(
    "../input/cropped-spects/test",
    shuffle=True,    target_size=(216,220),
    class_mode='categorical',
    batch_size=32
)


# In[ ]:


#in order to save the best model
checkpoint = ModelCheckpoint('model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
                             verbose=2, monitor='val_loss',save_best_only=True, mode='auto')


# In[ ]:


epochs=15

history = model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint]#using chekpoint we save the best model
)


# In[ ]:


model.evaluate(test_generator)


# In[ ]:


epochs=5

history2 = model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint]
    #validation_steps=total_validate/batch_size,
    #steps_per_epoch=total_train/batch_size,
)


# In[ ]:


epochs=5
history3 = model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint]
    #validation_steps=total_validate/batch_size,
    #steps_per_epoch=total_train/batch_size,
)


# In[ ]:


epochs=20
history4 = model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint]
    #validation_steps=total_validate/batch_size,
    #steps_per_epoch=total_train/batch_size,
)


# In[ ]:


#Graphs
plt.plot(history.history['loss']+history2.history['loss']+history3.history['loss'])
plt.plot(history.history['val_loss']+history2.history['val_loss']+history3.history['val_loss'])#+history4.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()


# In[ ]:



plt.plot(history.history['accuracy']+history2.history['accuracy']
         +history3.history['accuracy'])#+history4.history['accuracy'])
plt.plot(history.history['val_accuracy']+history2.history['val_accuracy']
         +history3.history['val_accuracy'])#+history4.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()


# In[ ]:


model.evaluate(test_generator)


# In[ ]:


os.remove("./model-011-0.685167-0.661000.h5")


# In[ ]:


best_model3=keras.models.load_model("./model-003-0.753667-0.664500.h5")
best_model3.summary()
best_model3.evaluate(    test_generator)


# In[ ]:


optimizer = keras.optimizers.Adam(lr=0.0001)
model2.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[ ]:


categories = []
filename_list=[]
for dirname, _, filenames in os.walk('../input/cropped-spects/test'):
    l2=[]
    d=dirname.split("/")[-1]
    categories.append(d)
    for filename in filenames:
        l2.append(dirname+"/"+filename)
    filename_list.extend(natsorted(l2))

    
del categories[0]
labels=[i for i in categories for j in range(200)]


# In[ ]:


df_test = pd.DataFrame({
    'filename': filename_list,
    'category': labels
})


# In[ ]:


categories_train = []
filename_list_train=[]
for dirname, _, filenames in os.walk('../input/cropped-spects/train'):
    l2=[]
    d=dirname.split("/")[-1]
    categories_train.append(d)
    for filename in filenames:
        l2.append(dirname+"/"+filename)
    filename_list_train.extend(natsorted(l2))
#del filename_list[0]
del categories_train[0]
labels_train=[i for i in categories_train for j in range(600)]

df_train=pd.DataFrame({
    'filename': filename_list_train,
    'category': labels_train
})


# In[ ]:


categories_val = []
filename_list_val=[]
for dirname, _, filenames in os.walk('../input/cropped-spects/validation'):
    l2=[]
    d=dirname.split("/")[-1]
    categories_val.append(d)
    for filename in filenames:
        l2.append(dirname+"/"+filename)
    filename_list_val.extend(natsorted(l2))
#del filename_list[0]
del categories_val[0]
labels_val=[i for i in categories_val for j in range(200)]

df_val=pd.DataFrame({
    'filename': filename_list_val,
    'category': labels_val
})


# In[ ]:


label_map = (train_generator.class_indices)
label_map

d2={}#swap key value pairs
for k,v in label_map.items():
    d2[v]=k
print(d2,d2.values())


# In[ ]:


def get_pred_list(model,df):
    freq=np.zeros((1,10))
    pred_list=list()
    for idx,row in tqdm(df.iterrows()):
        image = keras.preprocessing.image.load_img(row.filename,target_size=(216,220))
        input_arr = keras.preprocessing.image.img_to_array(image)/(255.)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        predictions = model.predict(input_arr)
        freq+=predictions
        if (idx+1)%10==0:
            pred_list.append(np.argmax(freq))
            freq=0
    return pred_list


# In[ ]:


pred_list=get_pred_list(df_train)
pred_list_val=get_pred_list(df_val)


# In[ ]:


pred_list_test=get_pred_list(df_test)


# In[ ]:


def acc_calculator(model,label_map,df):
    pred_list=get_pred_list(model,df)
    key_list = list(label_map.keys())
    val_list = list(label_map.values())
    counter = 0

    acc_counter = 0
    label_list=list()
    key_list_pos=list()

    for idx,row in tqdm(df.iterrows()):
        if (idx+1)%10 == 0:
            label = row.category
            label_list.append(label)            
            position = val_list.index(pred_list[counter])
            counter += 1
            key_list_pos.append(key_list[position])
            if label == key_list[position]:
                
                acc_counter += 1
    return acc_counter,acc_counter/len(pred_list),key_list_pos,label_list


# In[ ]:


#model3=keras.models.load_model("../input/model4/model-004-0.891667-0.712500.h5")
acc_counter,acc,key_list_pos,label_list=acc_calculator(best_model3,label_map,df_test)#for train 
print(acc_counter)


# In[ ]:


cm = confusion_matrix(label_list, key_list_pos)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=d2.values())
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Confusion Matrix of CNN',fontdict={'size':'21',"color":"red"})
disp.plot(values_format=".3g",ax=ax,xticks_rotation=45)


# In[ ]:



cm = confusion_matrix(label_list, key_list_pos)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=d2.values())
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Confusion Matrix of CNN',fontdict={'size':'21',"color":"red"})
disp.plot(values_format=".3g",ax=ax,xticks_rotation=45)


# In[ ]:


acc_counter,_accuracy=acc_calculator(best_model3,label_map,df_train)#for train 
print(acc_counter,_accuracy)
acc_counter=acc_calculator(best_model3,label_map,df_val)#for train 
print(acc_counter)
acc_counter=acc_calculator(best_model3,label_map,df_test)#for train 
print(acc_counter)


# In[ ]:


feature_extractor = Sequential()
for layer in best_model3.layers[:-5]:
    feature_extractor.add(layer)
feature_extractor.summary()


# In[ ]:


feature_extractor.add(keras.layers.GlobalAveragePooling2D())


# In[ ]:


feature_extractor.add(Flatten())
feature_extractor.summary()


# In[ ]:


def feature_extraction(df):
    feat_all_list=list()
    feat_list=list()
    for idx,row in tqdm(df.iterrows()):
        image = keras.preprocessing.image.load_img(row.filename,target_size=(216,220))
        input_arr = keras.preprocessing.image.img_to_array(image)/(255.)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        predictions = feature_extractor.predict(input_arr)
        feat_list.extend(predictions)
        if (idx+1)%10==0:
            feat_all_list.append(feat_list)
            feat_list=list()
    return np.array(feat_all_list).reshape((-1,1280))


# In[ ]:


test_feats=feature_extraction(df_test)
train_feats=feature_extraction(df_train)
val_feats=feature_extraction(df_val)
#extracted features are saved for traditional machine learning algorithms


# In[ ]:


np.savetxt("test_feats.csv", test_feats, delimiter=",")
np.savetxt("train_feats.csv", train_feats, delimiter=",")
np.savetxt("val_feats.csv", val_feats, delimiter=",")


# In[ ]:


model.save("./kaggle/xx")

