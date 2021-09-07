__all__ = ['Pix2PixLoss']


import tensorflow as _tf


_loss = _tf.keras.losses.BinaryCrossentropy(from_logits=True)
class Pix2PixLoss:
    """Class, describes generator and discriminator losses"""
    def __init__(self, LAMBDA=100):
        self.LAMBDA = LAMBDA


    def generator_loss(self, disc_desicions, generated, target):
        """Returns tuple of (adv_loss, l1_loss, total_loss) by given inputs:
        disc_desicions, generated, target"""

        ones = _tf.ones_like(disc_desicions)
        adv_loss = _loss(ones, disc_desicions)

        l1_loss = _tf.reduce_mean(_tf.abs(target - generated))
        total_loss = adv_loss + l1_loss * self.LAMBDA

        return adv_loss, l1_loss, total_loss


    def discriminator_loss(self, disc_desicions_fake, disc_desicions_real):
        """Returns binary crossentropy for generator"""

        return _loss(_tf.zeros_like(disc_desicions_fake), disc_desicions_fake) + _loss(_tf.ones_like(disc_desicions_real), disc_desicions_real)