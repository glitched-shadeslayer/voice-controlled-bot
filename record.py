# import the needed libraries
import pyaudio
import wave
import numpy as np
from matplotlib import pyplot as plt
import random
import soundfile as sf
from python_speech_features import mfcc
# Library for machine learning
import keras
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import Sequential,Model
from keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Embedding,LSTM,GaussianNoise, GRU
from keras.layers import Conv1D,SimpleRNN,MaxPooling1D,GlobalAveragePooling1D
import librosa
from librosa.feature import mfcc 
import keras.backend as K
import tensorflow as tf
######################################
# this library is to load the model and use it for testing
from keras.models import load_model
######################################


CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 1
RATE = 8000 #sample rate
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "testab.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK) #buffer
print("* recording")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data) # 2 bytes(16 bits) per channel
print("* done recording")
stream.stop_stream()
stream.close()
p.terminate()
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
b =WAVE_OUTPUT_FILENAME

data, samplerate = sf.read(b)
x= len(data)
p = 25000-x
l =0
tests = np.empty([200,4043])
new_data = np.empty([25000,])
y1 = np.empty([25000,])	

y = p//2;

for i in range(0,y):
    new_data[i] = y1 [i]
for i in range(y,x+y):
    new_data[i] = data[i-y]
for i in range(x+y,25000):
    new_data[i] = y1[i]

data = (mfcc(y = new_data, sr = samplerate, n_mfcc=39).T)
data = data.reshape((1,data.shape[0],data.shape[1]))
print(data.shape)
nIn = 4043
nOut = 5

########################################################################################

# Def Loss function

# this is the implementation of categorical loss function from scratch
# the input to the function is predicted probability and a one hot vector
# it computes the loss by summing over the expression -ylog(p) over the tuple
# this is summed over a batch and the final loss is given
def categorical_cross_entropy(ytrue, ypred, axis=-1):
    return -1.0*tf.reduce_mean(tf.reduce_sum(ytrue * tf.log(ypred), axis))
########################################################################################
model = load_model('model.h5', custom_objects={'categorical_cross_entropy':categorical_cross_entropy})

pred = model.predict(data)

pred = np.argmax(pred)

if(pred==0):
    print (pred,'back')
if(pred==1):
    print (pred,'forward')
if(pred==2):
    print (pred,'left')
if(pred==3):
    print (pred,'right')
if(pred==4):
    print (pred,'stop')

