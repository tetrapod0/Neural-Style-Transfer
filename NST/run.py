#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import time
import glob
import sys

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


target_img_path = glob.glob('./content_img/*.jpg')[0]
style_img_path = glob.glob('./style_img/*.jpg')[0]
IMG_MAX_SIZE = 512
style_weight = 1e-2
content_weight = 1e4
LEN_STYLE = 5
LAYERS = ['block1_conv1',
          'block2_conv1',
          'block3_conv1', 
          'block4_conv1', 
          'block5_conv1', 
          'block5_conv4']

epochs = int(sys.argv[1])


def load_img(img_path):
    img = tf.image.decode_jpeg(tf.io.read_file(img_path)) # uint8
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:2], tf.float32)
    scale = IMG_MAX_SIZE / max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
    new_img = tf.image.resize(img, new_shape) # float
    return new_img[tf.newaxis, ...]

def vgg_layers():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in LAYERS]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(output1):
    res = tf.einsum('abcd,abce->ade', output1, output1)
    shape = tf.shape(output1)
    divisor = tf.cast(shape[1] * shape[2], tf.float32)
    return res[0] / divisor


target_img = load_img(target_img_path)
style_img = load_img(style_img_path)

model = vgg_layers()

pre_style_img = tf.keras.applications.vgg19.preprocess_input(style_img * 255.0)
pre_target_img = tf.keras.applications.vgg19.preprocess_input(target_img * 255.0)
style_outputs = model(pre_style_img)
target_outputs = model(pre_target_img)

style_outputs = list(map(gram_matrix, style_outputs))
target_outputs = list(map(gram_matrix, target_outputs))

goal_outputs = style_outputs[:LEN_STYLE] + target_outputs[LEN_STYLE:]


def get_loss(outputs, goal_outputs):
    style_loss = tf.add_n([tf.reduce_mean((out - goal)**2) for out, goal in zip(outputs[:LEN_STYLE], goal_outputs[:LEN_STYLE])])
    content_loss = tf.add_n([tf.reduce_mean((out - goal)**2) for out, goal in zip(outputs[LEN_STYLE:], goal_outputs[LEN_STYLE:])])
    
    style_loss *= style_weight / LEN_STYLE
    content_loss *= content_weight / (len(LAYERS) - LEN_STYLE)

    total_loss = style_loss + content_loss

    return total_loss


image = tf.Variable(target_img)

optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

def train_step():
    with tf.GradientTape() as tape:
        tape.watch(image)
        outputs = tf.keras.applications.vgg19.preprocess_input(image * 255.0)
        outputs = model(outputs)
        outputs = list(map(gram_matrix, outputs))

        loss = get_loss(outputs, goal_outputs)
        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, 0.0, 1.0))


def run():
    start = time.time()
    step = 0
    for epoch in range(epochs):
        for m in range(50):
            step += 1
            train_step()
        tf.keras.preprocessing.image.save_img('./result_img/image.jpg', image[0])
        print("Train step: {}".format(step))
        with open('./temp.txt', 'w') as f:
            f.write('{}'.format(epoch+1))
            
            
    end = time.time()
    print("Total time: {:.1f}".format(end-start))



run()






