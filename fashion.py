import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from random import randint

# Load datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

def show_random_image():
    randomIndex = randint(0,x_train.shape[0]-1)
    plt.imshow(x_train[randomIndex])
    print('Showing image at index %i.' % randomIndex)

if __name__ == '__main__':
    pass
