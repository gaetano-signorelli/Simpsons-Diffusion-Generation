import tensorflow as tf
from tensorflow.keras import layers, activations

class ConvolutionalBlockLayer(layers.Layer):

    def __init__(self, out_channels, mid_channels=None, residual=False):

        super(ConvolutionalBlockLayer).__init__()

        self.residual = residual

        if not mid_channels:
            mid_channels = out_channels

        self.conv_layer_1 = layers.Conv2d(mid_channels,
                                        kernel_size=3,
                                        padding="same",
                                        use_bias=False)

        self.conv_layer_2 = layers.Conv2d(out_channels,
                                        kernel_size=3,
                                        padding="same",
                                        use_bias=False)

        self.group_norm_layer_1 = layers.BatchNormalization(axis=(0,3))
        self.group_norm_layer_2 = layers.BatchNormalization(axis=(0,3))

    @tf.function
    def call(self, x):

        conv_x = self.conv_layer_1(x)
        conv_x = self.group_norm_layer_1(conv_x)
        conv_x = activations.gelu(conv_x)
        conv_x = self.conv_layer_2(conv_x)
        conv_x = self.group_norm_layer_2(conv_x)

        if self.residual:
            return activations.gelu(x + conv_x)

        else:
            return conv_x
