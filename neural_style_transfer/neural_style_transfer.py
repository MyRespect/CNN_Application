import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


def prepare_image(style_img_path='./images/claude-monet.jpg', content_img_path='./images/louvre_small.jpg', target_size=(225, 300), noise_ratio=0.4):
    style_img = image.load_img(style_img_path, target_size=target_size)
    style_img = np.array(style_img)
    style_img = style_img[tf.newaxis, ...]
    style_img = tf.cast(style_img, tf.float32)

    content_image = image.load_img(content_img_path, target_size=target_size)
    content_image = np.array(content_image)
    content_image = content_image[tf.newaxis, ...]
    content_image = tf.cast(content_image, tf.float32)

    noise_image = np.random.uniform(-20, 20, (1,) + target_size + (3,))
    input_image = tf.Variable(
        noise_image * noise_ratio + content_image * (1 - noise_ratio), dtype=tf.float32)
    # input_image = tf.Variable(np.random.rand(1, 225, 300, 3), dtype=tf.float32)

    # print(input_image)
    out_img = input_image[0]
    image2 = np.clip(out_img, 0, 255).astype('uint8')
    Image.fromarray(image2).save('./style2.png')

    return (style_img, content_image, input_image)


def prepare_model(img_shape=(225, 300, 3)):
    model = VGG16(include_top=False, weights='imagenet', input_shape=img_shape)
    block1_conv1 = Model(inputs=model.input,
                         outputs=model.get_layer('block3_conv1').output)
    block2_conv2 = Model(inputs=model.input,
                         outputs=model.get_layer('block3_conv2').output)
    block3_conv3 = Model(inputs=model.input,
                         outputs=model.get_layer('block3_conv3').output)
    block4_conv1 = Model(inputs=model.input,
                         outputs=model.get_layer('block4_conv1').output)
    block4_conv2 = Model(inputs=model.input,
                         outputs=model.get_layer('block4_conv2').output)
    block5_conv3 = Model(inputs=model.input,
                         outputs=model.get_layer('block5_conv1').output)
    layers = [block1_conv1, block2_conv2, block3_conv3,
              block4_conv1, block4_conv2, block5_conv3]
    coeff = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
    return layers, coeff


prepare_model()


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G
    a_C_unrolled = tf.reshape(a_C, [n_C, -1])
    a_G_unrolled = tf.reshape(a_G, [n_C, -1])

    J_content = 1. / (4 * n_H * n_W * n_C) * \
        tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content


def compute_style_cost(img_style, img_gen, layers, coeff):
    J_style = 0
    for i, layer in enumerate(layers):
        a_S = layer(img_style) #problems with layer.predict(img_gen) since it requires np.array
        a_G = layer(img_gen)
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff[i] * J_style_layer
    return J_style


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    def gram_matrix(A):
        GA = tf.matmul(A, tf.transpose(A))
        return GA

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, (-1, n_C)))
    a_G = tf.transpose(tf.reshape(a_G, (-1, n_C)))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = 1. / (4 * n_C * n_C * n_W * n_W * n_H * n_H) * \
        tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

    return J_style_layer


if __name__ == "__main__":
    imgs = prepare_image()
    layers, coeff = prepare_model()
    optimizer = tf.keras.optimizers.Adam(1.0)

    @tf.function
    def train_step(imgs, layers, coeff, num=3, alpha=10, beta=40):
        with tf.GradientTape() as tape:
            a_C = layers[num](imgs[1])
            a_G = layers[num](imgs[2])

            J_content = compute_content_cost(a_C, a_G)
            J_style = compute_style_cost(imgs[0], imgs[2], layers, coeff)
            loss = alpha * J_content + beta * J_style
        gradients = tape.gradient(loss, [imgs[2]])
        optimizer.apply_gradients(zip(gradients, [imgs[2]]))

    for i in range(150):
        train_step(imgs, layers, coeff)

    print(np.shape(imgs[2]))
    out_img = imgs[2][0]

    image = np.clip(out_img, 0, 255).astype('uint8')
    Image.fromarray(image).save('./style.png')
