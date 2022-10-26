import tensorflow as tf
from tensorflow.keras import Model

from src.model.unet import UNet

class Diffusion(Model):

    def __init__(self, input_shape, n_heads, time_embedding_size,
                beta_start, beta_end, noise_steps):

        super(Diffusion).__init__()

        self.input_shape = input_shape

        self.unet_model = UNet(input_shape, n_heads, time_embedding_size)

        self.beta, self.alpha, self.alpha_hat = self.get_linear_schedules(beta_start,
                                                                        beta_end,
                                                                        noise_steps)

    def get_linear_schedules(self, start, end, steps):

        beta = tf.linspace(start, end, steps)
        alpha = 1.0 - beta
        alpha_hat = tf.math.cumprod(alpha)

        return beta, alpha, alpha_hat

    @tf.function
    def noise_images(self, x, timesteps):

        gathered_alpha_hat = tf.gather(self.alpha_hat, timesteps)
        sqrt_alpha_hat = tf.math.sqrt(gathered_alpha_hat)
        sqrt_alpha_hat = tf.expand_dims(sqrt_alpha_hat, axis=-1)
        sqrt_alpha_hat = tf.expand_dims(sqrt_alpha_hat, axis=-1)
        sqrt_alpha_hat = tf.expand_dims(sqrt_alpha_hat, axis=-1)

        eps = tf.random.normal(self.input_shape)

        gaussian_noise = sqrt_alpha_hat * x + (1-sqrt_alpha_hat) * eps

        return gaussian_noise, eps

    @tf.function
    def call(self, x):
        pass
