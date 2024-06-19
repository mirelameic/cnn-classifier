import tensorflow as tf
from tensorflow import keras
from keras import datasets

def load_and_preprocess_data():
    # carrega o conjunto de dados MNIST
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # normaliza as imagens para o intervalo [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # adiciona uma dimensão de canal para as imagens
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]

    return (train_images, train_labels), (test_images, test_labels)

def filter_binary_classes(images, labels, class_1, class_2):
    # filtra as imagens e rótulos para obter apenas as duas classes escolhidas
    binary_mask = (labels == class_1) | (labels == class_2)
    binary_images = images[binary_mask]
    binary_labels = labels[binary_mask]
    return binary_images, binary_labels
