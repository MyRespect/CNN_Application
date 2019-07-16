import os
import glob
import time
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential([
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(filters=32, kernel_size=3,
                          strides=(2, 2), activation='relu'),
            layers.Conv2D(filters=64, kernel_size=3,
                          strides=(2, 2), activation='relu'),
            layers.Flatten(),
            # No activation
            layers.Dense(latent_dim + latent_dim)
        ])
        self.generative_net = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(units=7 * 7 * 32, activation='relu'),
            layers.Reshape(target_shape=(7, 7, 32)),
            layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(
                2, 2), padding='SAME', activation='relu'),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(
                2, 2), padding='SAME', activation='relu'),
            layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding='SAME')
        ])

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(
            x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2 * np.pi)
    return tf.reduce_sum(-0.5 * ((sample - mean)**2 * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


def draw_gif(anim_file='cvae.gif'):
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    import IPython
    if IPython.version_info >= (6, 2, 0, ''):
        display.Image(filename=anim_file)


def prepare_data():
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(
        test_images.shape[0], 28, 28, 1).astype('float32')

    # normalizing the images from [0, 255] to [0, 1]
    train_images /= 255.
    test_images /= 255.

    # Binarization, model each pixel with a Bernoulli distribution
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.

    TRAIN_BUF = 60000
    BATCH_SIZE = 128

    TEST_BUF = 10000

    train_data = tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_data = tf.data.Dataset.from_tensor_slices(
        test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

    return train_data, test_data


if __name__ == "__main__":
    epochs = 100
    latent_dim = 50
    num_examples_to_generate = 16

    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    model = CVAE(latent_dim)

    train_dataset, test_dataset = prepare_data()

    for epoch in range(epochs):
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x)
        end_time = time.time()

        if epoch % 10 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(model, test_x))
            elbo = -loss.result()
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, ''time elapse for current epoch {}'.format(
                epoch, elbo, end_time - start_time))
            generate_and_save_images(
                model, epoch, random_vector_for_generation)
