__all__ = ['Pix2PixGAN', 'losses']

import tensorflow as _tf
import numpy as _np
from . import losses



class Pix2PixGAN:
    def __init__(self, generator:_tf.keras.Model, discriminator:_tf.keras.Model,
                 generator_optimizer:_tf.keras.optimizers.Optimizer=None, discriminator_optimizer:_tf.keras.optimizers.Optimizer=None,
                 loss:losses.Pix2PixLoss=losses.Pix2PixLoss()):
        
        self.generator = generator
        self.discriminator = discriminator


        if generator_optimizer is None:
            generator_optimizer = _tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        if discriminator_optimizer is None:
            discriminator_optimizer = _tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.loss = loss


    
    def fit(self, datagenerator, iterations, callback=None, callback_period=100):
        datagenerator.reset()

        @_tf.function
        def train_step(input_image, target):
            with _tf.GradientTape() as gen_tape, _tf.GradientTape() as disc_tape:
                generated = self.generator(input_image, training=True)

                disc_real = self.discriminator([input_image, target], training=True)
                disc_fake = self.discriminator([input_image, generated], training=True)

                adv_loss, l1_loss, gen_loss = self.loss.generator_loss(disc_fake, generated, target)
                disc_loss = self.loss.discriminator_loss(disc_fake, disc_real)

            gen_gradient = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            disc_gradient = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(disc_gradient, self.discriminator.trainable_variables))

            return adv_loss, l1_loss, gen_loss, disc_loss
        
        for i in range(iterations):
            x, y = next(datagenerator)

            if callback is not None and i % callback_period == 0:
                callback(i)


            adv_loss, l1_loss, gen_loss, disc_loss = train_step(x, y)

            if i % 100 == 0:
                print(f'{i}: adv_loss: {_np.round(adv_loss, 4)} \t l1_loss: {_np.round(l1_loss, 4)}' \
                    f'\t gen_loss: {_np.round(gen_loss, 4)} \t disc_loss: {_np.round(disc_loss, 4)}')