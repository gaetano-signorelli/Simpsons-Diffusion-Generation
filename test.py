from src.model.layers.positional_encoding import PositionalEncodingLayer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

pos_layer = PositionalEncodingLayer(512)

tensor = tf.range(100)
tensor = tf.reshape(tensor, (100,1))

encoding = pos_layer(tensor).numpy()

cax = plt.matshow(encoding)
plt.gcf().colorbar(cax)
plt.show()
