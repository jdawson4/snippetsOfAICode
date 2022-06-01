# This is an implementation of the keras documentation's
# Description of a Conditional Generative Adverserial Network (CGAN)
# The walkthrough/example can be found here: https://keras.io/examples/generative/conditional_gan/
# 
# This file will likely be a copy-paste of the code found there, but I want to experiment
# and experience what it's like working with this model in order to be able to play around with
# more advanced versions of it.

# imports:
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio

# constants and hyperparameters
batch_size = 64
num_channels = 1
num_classes = 10 # we're using MNIST
image_size = 28
latent_dim = 128

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# scale to [0,1], add channel dim, and one-hot encode the labels
all_digits = all_digits.astype("float32")/255.0
all_digits = np.reshape(all_digits,(-1,28,28,1))
all_labels = keras.utils.to_categorical(all_labels,10)

# create a tensorflow dataset object
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size = 1024).batch(batch_size)

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")

# calculate input channels for g and d
generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

# the good stuff:

# create discriminator
discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((28,28, discriminator_in_channels)),
        layers.Conv2D(64, (3,3), strides = (2,2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3,3), strides=(2,2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name='discriminator',
)

# create generator
generator = keras.Sequential(
    [
        keras.layers.InputLayer((generator_in_channels,)),
        layers.Dense(7 * 7 * generator_in_channels),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, generator_in_channels)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name='generator',
)

# creating a ConditionalGAN model (exciting!)
class ConditionalGAN(keras.Model):
    def __init__(self,discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name='gnerator_loss')
        self.disc_loss_tracker = keras.metrics.Mean(name='discriminator_loss')

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
    
    def train_step(self,data):
        # unpack data
        real_images, one_hot_labels = data

        # add dummy dimensions to the labels so that they can be concatenated with the images.
        # this is for the discriminator.
        image_one_hot_labels = one_hot_labels[:,:,None,None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # sample random points in the latent space and concatenate the labels
        # this is for the generator
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # decode the noise (guided by labels) to fake images
        generated_images = self.generator(random_vector_labels)
        # this line actually does the computation i guess

        # combine with the real images. Note that we are concatenating the labels with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size,1))], axis=0
        )

        # train discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images) # actually does the computation
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights) # feels like this should be indented???
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights) # this actually does the backprop, I think?
        )

        # sample more random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # assemble labels that say "all real images" (????)
        misleading_labels = tf.zeros((batch_size, 1))

        # train the generator! (not: NOT updating the weights of discriminator, we've already done that.)
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels) # make generative model do work.
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels) # make discriminative model predict on fakes
            g_loss = self.loss_fn(misleading_labels, predictions) # determine how good g was at fooling d
        grads = tape.gradient(g_loss, self.generator.trainable_weights) # SGD
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights)) # backprop on generator

        # monitor loss
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)

        # and finish!
        return {
            'g_loss': self.gen_loss_tracker.result(),
            'd_loss': self.disc_loss_tracker.result(),
        }

# instantiate and train
cond_gan=ConditionalGAN(
    discriminator = discriminator, generator=generator, latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer = keras.optimizers.Adam(learning_rate = 0.0003),
    g_optimizer = keras.optimizers.Adam(learning_rate = 0.0003),
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True),
)

cond_gan.fit(dataset,epochs = 20)

# We first extract the trained generator from our Conditiona GAN.
trained_gen = cond_gan.generator

# Choose the number of intermediate images that would be generated in
# between the interpolation + 2 (start and last images).
num_interpolation = 9  # @param {type:"integer"}

# Sample noise for the interpolation.
interpolation_noise = tf.random.normal(shape=(1, latent_dim))
interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, latent_dim))


def interpolate_class(first_number, second_number):
    # Convert the start and end labels to one-hot encoded vectors.
    first_label = keras.utils.to_categorical([first_number], num_classes)
    second_label = keras.utils.to_categorical([second_number], num_classes)
    first_label = tf.cast(first_label, tf.float32)
    second_label = tf.cast(second_label, tf.float32)

    # Calculate the interpolation vector between the two labels.
    percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
    percent_second_label = tf.cast(percent_second_label, tf.float32)
    interpolation_labels = (
        first_label * (1 - percent_second_label) + second_label * percent_second_label
    )

    # Combine the noise and the labels and run inference with the generator.
    noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
    fake = trained_gen.predict(noise_and_labels)
    return fake


start_class = 1  # @param {type:"slider", min:0, max:9, step:1}
end_class = 5  # @param {type:"slider", min:0, max:9, step:1}

fake_images = interpolate_class(start_class, end_class)
fake_images *= 255.0
converted_images = fake_images.astype(np.uint8)
converted_images = tf.image.resize(converted_images, (96, 96)).numpy().astype(np.uint8)
imageio.mimsave("animation.gif", converted_images, fps=1)
#embed.embed_file("animation.gif") # I think this is for colab???

#cond_gan.summary() # I just wanna see if this works

# it works! I will say, though, the digits it created are... bad.
# Maybe a W-GAN would do better? Maybe more epochs? Either way,
# # a successful experiment!