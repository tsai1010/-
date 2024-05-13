import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import os
import sklearn
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

wav_dir = '/content/drive/MyDrive/piano_data/piano_train'
wavs = [os.path.join(wav_dir, wav) for wav in os.listdir(wav_dir)]

def preprocess(wav):
  wav = tfio.audio.decode_wav(tf.io.read_file(wav), dtype=tf.int32)
  wav = tf.pad(wav, tf.constant([[0,200000], [0,0]]), 'CONSTANT')
  wav = wav[:262144]

  wav = wav/(2**16)
  wav = tf.cast(wav, tf.float32)
  # wav = tfio.audio.resample(wav, 48000, 24000)
  wav = wav/32768.
  wav = tf.signal.stft(wav[:,0], frame_length=1024, frame_step=512, pad_end=False, window_fn=tf.signal.hann_window)
  wav = 2*np.abs(s)/np.sum(256)
  
  return wav

def get_label(wav):
  wav_str = wav.replace('v', '_')
  wav_str = wav_str.split('_')
  wav_str = wav_str[1]
  return wav_str

# 定義標籤字典
label_dict = {
    0: "A0",
    1: "C1",    2: "D#1",    3: "F#1",    4: "A1",
    5: "C2",    6: "D#2",    7: "F#2",    8: "A2",
    9: "C3",    10: "D#3",   11: "F#3",   12: "A3",
    13: "C4",   14: "D#4",   15: "F#4",   16: "A4",
    17: "C5",   18: "D#5",   19: "F#5",   20: "A5",
    21: "C6",   22: "D#6",   23: "F#6",   24: "A6",
    25: "C7",   26: "D#7",   27: "F#7",   28: "A7",
    29: "C8",
}

# 建立label
labelencoder = LabelEncoder()
wav_label = list(map(get_label, wavs))
wav_label = tf.convert_to_tensor(wav_label)
wav_label = labelencoder.fit_transform(wav_label)
wav_label = keras.utils.to_categorical(wav_label)

# 建立dataset
data = tf.data.Dataset.from_tensor_slices((wavs))
wav_data = data.map(preprocess)
wav_data = list(wav_data) #資料太多會掛
wav_data = np.array(wav_data)

# 測試用
w = preprocess(wavs[50])
print('w:', w.shape)
s = tf.signal.stft(w[:,0], frame_length=1024, frame_step=512, pad_end=False, window_fn=tf.signal.hann_window)
s = 2*np.abs(s)/np.sum(256)
print('s:', s.shape)
np.max(s)

from keras.layers import Dense, Input, Flatten, BatchNormalization, LeakyReLU, Conv1D, Conv1DTranspose, Reshape
from keras.models import Sequential

# 建立模型
input_wav = keras.Input(shape=(511,513,1))

x = Conv2D(8, (5,5), strides=1, activation='relu', padding='same')(input_wav)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = BatchNormalization()(x)

x = Conv2D(16, (5,5), strides=1, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = BatchNormalization()(x)

x = Conv2D(32, (5,5), strides=1, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = BatchNormalization()(x)

x = Conv2D(32, (3,3), strides=1, activation='relu', padding='same')(x)
x = BatchNormalization()(x)

x = Conv2D(32, (3,3), strides=1, activation='relu', padding='same')(x)
x = BatchNormalization()(x)

x = Conv2D(32, (3,3), strides=1, activation='relu', padding='same')(x)
x = BatchNormalization()(x)

x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.65)(x)
x = BatchNormalization()(x)
encoder_output = Dense(30, activation='softmax')(x)
encoder = keras.Model(input_wav, encoder_output)
encoder.summary()

encoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-6),
    loss='categorical_crossentropy',
    metrics=['acc']
)

encoder.load_weights("/content/piano_model.h5")
train_history = encoder.fit(wav_data, wav_label, batch_size=16, epochs=20, verbose=1, shuffle=True)

try:
  encoder.save_weights("/content/piano_model.h5")
  print("success")
except:
  print("error")

# 獲取資料
wav_dir = '/content/drive/MyDrive/piano_data/piano_test'
wavs = [os.path.join(wav_dir, wav) for wav in os.listdir(wav_dir)]

# 建立test label
labelencoder = LabelEncoder()
test_label = list(map(get_label, wavs))
test_label = tf.convert_to_tensor(test_label)
test_label = labelencoder.fit_transform(test_label)
test_label = keras.utils.to_categorical(test_label)

# test data 前處理
data = tf.data.Dataset.from_tensor_slices((wavs))
test_data = data.map(preprocess)
test_data = list(test_data)
test_data = np.array(test_data)

# 載入模型
model = encoder
model.load_weights("/content/drive/MyDrive/piano_model.h5")
# 進行預測
predictions = model.predict(test_data)

# 計算混淆矩陣
test_true = [label_dict[np.argmax(y)] for y in test_label]
test_pred = [label_dict[np.argmax(pred)] for pred in model.predict(test_data)]
cm = confusion_matrix(test_true, test_pred, labels=list(label_dict.values()))

# 繪製混淆矩陣
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_dict.values()), yticklabels=list(label_dict.values()))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import accuracy_score

# 計算測試集準確率
accuracy = accuracy_score(test_true, test_pred)
print("Test Accuracy:", accuracy * 100, "%")
