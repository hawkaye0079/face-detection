# -*- coding: utf-8 -*-
"""
Created on Wed May 14 01:01:52 2025

@author: HARSHIT NARAIN
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

train_dir = "data/images/train"
val_dir = "data/images/val"
img_size = 224
batch_size = 16

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='binary')

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='binary')

if train_gen.samples == 0 or val_gen.samples == 0:
    raise ValueError("Training or validation dataset is empty! Please check your image folders.")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=preds)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

os.makedirs("model", exist_ok=True)
checkpoint = ModelCheckpoint('model/image_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

model.fit(train_gen, validation_data=val_gen, epochs=5, callbacks=[checkpoint])
