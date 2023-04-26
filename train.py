import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

label_map = {1: 'Droga z pierwszenstwem',
             2: 'Ustap pierwszenstwa',
             3: 'Stop',
             4: 'Zakaz wjazdu',
             5: 'Zakaz ruchu w obu kierunkach',
             6: 'Zakaz zatrzymywania sie',
             7: 'Zakaz wyprzedzania',
             8: 'Przejscie dla pieszych',
             9: 'Dzieci',
             10: 'Ograniczenie predkosci do 20',
             11: 'Ograniczenie predkosci do 30',
             12: 'Ograniczenie predkosci do 40',
             13: 'Ograniczenie predkosci do 50',
             14: 'Ograniczenie predkosci do 60',
             15: 'Ograniczenie predkosci do 70',
             16: 'Ograniczenie predkosci do 80',
             17: 'Ograniczenie predkosci do 90',
             18: 'Ograniczenie predkosci do 100',
             19: 'Ograniczenie predkosci do 120',
             20: 'Zielone swiatlo',
             21: 'Zolte swiatlo',
             22: 'Czerwone swiatlo'} 

train_images_dir = 'dataset\\train'
train_annotations_dir = 'dataset\\train_labels'
val_images_dir = 'dataset\\valid'
val_annotations_dir = 'dataset\\valid_labels'
test_images_dir = 'dataset\\test'
test_annotations_dir = 'dataset\\test_labels'

train_data = object_detector.DataLoader.from_pascal_voc(train_images_dir, train_annotations_dir, label_map=label_map)
validation_data = object_detector.DataLoader.from_pascal_voc(val_images_dir, val_annotations_dir, label_map=label_map)
test_data = object_detector.DataLoader.from_pascal_voc(test_images_dir, test_annotations_dir, label_map=label_map)

print("\n")
print(f'Train count: {len(train_data)}')
print(f'Validation count: {len(validation_data)}')
print(f'Test count: {len(test_data)}')
print("\n")

print("Training...")

spec = model_spec.get('efficientdet_lite1')
model = object_detector.create(train_data, model_spec=spec, batch_size=8, epochs=50, train_whole_model=True, validation_data=validation_data)
model.evaluate(test_data)
model.export(export_dir='model')
model.evaluate_tflite('model\\model.tflite', test_data)
model.summary()