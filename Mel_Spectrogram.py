#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import scipy
from scipy import misc
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import librosa
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, 
                          Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D)
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
import shutil
from keras.preprocessing.image import ImageDataGenerator
import random


# In[13]:


os.makedirs('D:\\ML Proje\\content')


# In[ ]:


os.makedirs('./content/spectrograms3sec')
os.makedirs('./content/spectrograms3sec/train')
os.makedirs('./content/spectrograms3sec/test')

genres = 'blues classical country disco pop hiphop metal reggae rock jazz'
genres = genres.split()
for g in genres:
  path_audio = os.path.join('D:\\ML Proje\\content\\audio3sec',f'{g}')
  os.makedirs(path_audio)
  path_train = os.path.join('D:\\ML Proje\\content\\spectrograms3sec\\train',f'{g}')
  path_test = os.path.join('D:\\ML Proje\\content\\spectrograms3sec\\test',f'{g}')
  os. makedirs(path_train)
  os. makedirs(path_test)


# In[6]:


from pydub import AudioSegment
i = 0
for g in genres:
  j=0
  print(f"{g}")
  for filename in os.listdir(os.path.join('D:\\ML Proje\\archive\\Data\\genres_original',f"{g}")):

    song  =  os.path.join(f'D:\\ML Proje\\archive\\Data\\genres_original\\{g}',f'{filename}')
    j = j+1
    for w in range(0,10):
      i = i+1
      #print(i)
      t1 = 3*(w)*1000
      t2 = 3*(w+1)*1000
      newAudio = AudioSegment.from_wav(song)
      new = newAudio[t1:t2]
      new.export(f'D:\\ML Proje\\content\\audio3sec\\{g}\\{g+str(j)+str(w)}.wav', format="wav")


# In[18]:


for g in genres:
    j=0
    for filename in os.listdir(os.path.join('D:\\ML Proje\\archive\\Data\\genres_original',f"{g}")):

        song  =  os.path.join(f'D:\\ML Proje\\archive\\Data\\genres_original\\{g}',f'{filename}')
        j = j+1
        print(song)
        break


# In[8]:


genres = 'blues classical country disco pop hiphop metal reggae rock jazz'
genres = genres.split()


for g in genres:
  #os. makedirs(f'D:\\ML Proje\\content\\spectrograms3sec\\all\\{g}')
  j = 0
  t=0
  for filename in os.listdir(os.path.join('D:\\ML Proje\\content\\audio3sec',f"{g}")):
    song  =  os.path.join(f'D:\\ML Proje\\content\\audio3sec\\{g}',f'{filename}')
    j = j+1
    t+=1
    y,sr = librosa.load(song,duration=3)
    mels = librosa.feature.melspectrogram(y=y,sr=sr)
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
    plt.savefig(f'D:\\ML Proje\\content\\spectrograms3sec\\all\\{g}\\{g+str(j)}.png')


# In[ ]:





# In[2]:


music_list=os.listdir(os.path.join('D:\\ML Proje\\content\\audio3sec',f"{genres[5]}"))
j=0


# In[71]:


genres = 'blues classical country disco pop hiphop metal reggae rock jazz'
genres = genres.split()

for g in genres:

    os. makedirs(f'D:\\ML Proje\\content\\spectrograms3sec\\all\\{g}')
    j = 0
    for filename in os.listdir(os.path.join('D:\\ML Proje\\content\\audio3sec',f"{g}")):
        song  =  os.path.join(f'D:\\ML Proje\\content\\audio3sec\\{g}',f'{filename}')
        j = j+1
        y,sr = librosa.load(song,duration=3)

        mels = librosa.feature.melspectrogram(y=y,sr=sr)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
        plt.savefig(f'D:\\ML Proje\\content\\spectrograms3sec\\all\\{g}\\{g+str(j)}.png')


# In[4]:


genres = 'blues classical country disco pop hiphop metal reggae rock jazz'
genres = genres.split()


# In[36]:


directory = "D:\\ML Proje\\content\\spectrograms3sec\\all\\"
test_genres=[blues_test,classical_test,country_test,disco_test,pop_test,hiphop_test,metal_test,reggae_test,rock_test,jazz_test]
counter=0

for i in test_genres:
    for f in i:
        for t in f:
            shutil.move(directory + f"{genres[counter]}"+ "\\" + t,"D:\\ML Proje\\content\\spectrograms3sec\\test\\" + f"{genres[counter]}")
    counter+=1
    
    """for f in test_files:
        shutil.move(directory + f"{g}"+ "\\" + f,"D:\\ML Proje\\content\\spectrograms3sec\\test\\" + f"{g}")"""


# In[38]:


directory = "D:\\ML Proje\\content\\spectrograms3sec\\all\\"
validation_genres=[blues_validation,classical_validation,country_validation,disco_validation
              ,pop_validation,hiphop_validation,metal_validation,reggae_validation,rock_validation,jazz_validation]
counter=0

for i in validation_genres:
    for f in i:
        for t in f:
            shutil.move(directory + f"{genres[counter]}"+ "\\" + t,"D:\\ML Proje\\content\\spectrograms3sec\\validation\\" + f"{genres[counter]}")
    counter+=1
    
    """for f in test_files:
        shutil.move(directory + f"{g}"+ "\\" + f,"D:\\ML Proje\\content\\spectrograms3sec\\test\\" + f"{g}")"""


# In[15]:


blues_10 = [["blues"+str(x*10+y)+".png" for y in range(1,11) ] for x in range(0,100) ]
classical_10 = [["classical"+str(x*10+y)+".png" for y in range(1,11) ] for x in range(0,100) ]
country_10 = [["country"+str(x*10+y)+".png" for y in range(1,11) ] for x in range(0,100) ]
disco_10 = [["disco"+str(x*10+y)+".png" for y in range(1,11) ] for x in range(0,100) ]
hiphop_10 = [["hiphop"+str(x*10+y)+".png" for y in range(1,11) ] for x in range(0,100) ]
jazz_10 = [["jazz"+str(x*10+y)+".png" for y in range(1,11) ] for x in range(0,100) ]
metal_10 = [["metal"+str(x*10+y)+".png" for y in range(1,11) ] for x in range(0,100) ]
pop_10 = [["pop"+str(x*10+y)+".png" for y in range(1,11) ] for x in range(0,100) ]
reggae_10 = [["reggae"+str(x*10+y)+".png" for y in range(1,11) ] for x in range(0,100) ]
rock_10 = [["rock"+str(x*10+y)+".png" for y in range(1,11) ] for x in range(0,100) ]


# In[35]:


from sklearn.model_selection import train_test_split
blues_train,blues_test = train_test_split(blues_10, test_size=0.2, random_state=42)
classical_train,classical_test = train_test_split(classical_10, test_size=0.2, random_state=42)
country_train,country_test = train_test_split(country_10, test_size=0.2, random_state=42)
disco_train,disco_test = train_test_split(disco_10, test_size=0.2, random_state=42)
hiphop_train,hiphop_test = train_test_split(hiphop_10, test_size=0.2, random_state=42)
jazz_train,jazz_test = train_test_split(jazz_10, test_size=0.2, random_state=42)
metal_train,metal_test = train_test_split(metal_10, test_size=0.2, random_state=42)
pop_train,pop_test = train_test_split(pop_10, test_size=0.2, random_state=42)
reggae_train,reggae_test = train_test_split(reggae_10, test_size=0.2, random_state=42)
rock_train,rock_test = train_test_split(rock_10, test_size=0.2, random_state=42)


# In[37]:


blues_train,blues_validation = train_test_split(blues_train, test_size=0.25, random_state=42)
classical_train,classical_validation = train_test_split(classical_train, test_size=0.25, random_state=42)
country_train,country_validation = train_test_split(country_train, test_size=0.25, random_state=42)
disco_train,disco_validation = train_test_split(disco_train, test_size=0.25, random_state=42)
hiphop_train,hiphop_validation = train_test_split(hiphop_train, test_size=0.25, random_state=42)
jazz_train,jazz_validation = train_test_split(jazz_train, test_size=0.25, random_state=42)
metal_train,metal_validation = train_test_split(metal_train, test_size=0.25, random_state=42)
pop_train,pop_validation = train_test_split(pop_train, test_size=0.25, random_state=42)
reggae_train,reggae_validation = train_test_split(reggae_train, test_size=0.25, random_state=42)
rock_train,rock_validation = train_test_split(rock_train, test_size=0.25, random_state=42)


# In[27]:





# In[ ]:





# In[27]:


train_dir = "\\content\\spectrograms3sec\\train\\"
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(288,432),color_mode="rgba",class_mode='categorical',batch_size=128)

validation_dir = "\\content\\spectrograms3sec\\test\\"
vali_datagen = ImageDataGenerator(rescale=1./255)
vali_generator = vali_datagen.flow_from_directory(validation_dir,target_size=(288,432),color_mode='rgba',class_mode='categorical',batch_size=128)

