# -*- coding: utf-8 -*-
"""
Created on Tue May 13 23:28:52 2025

@author: HARSHIT NARAIN
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Load your dataset: sequences of features
X = np.load('data/X.npy')  # shape: (num_samples, timesteps, 2048)
y = np.load('data/y.npy')  # shape: (num_samples,)

model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint('model/lstm_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

model.fit(X, y, epochs=10, batch_size=8, validation_split=0.2, callbacks=[checkpoint])
