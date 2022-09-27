from model import SSD300
import os
import tensorflow as tf

def to_tflite(model, save_path='models/mobilenet_ssd.tflite'):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(save_path, "wb").write(tflite_model)

def to_quantized_tflite(model, save_path='models/mobilenet_ssd_quantized.tflite'):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open(save_path, "wb").write(tflite_model)

ssd = SSD300([300, 300, 3], 11)
ssd.load_weights('models/mobilenet_ssd.h5', by_name=True)
to_tflite(ssd)
to_quantized_tflite(ssd)
