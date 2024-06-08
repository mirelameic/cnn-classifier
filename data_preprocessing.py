import tensorflow as tf
from tensorflow.keras import datasets

def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # Normalizar as imagens
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Expandir as dimensões para adicionar o canal (necessário para o CNN)
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]

    return (train_images, train_labels), (test_images, test_labels)

def filter_binary_classes(images, labels, class_1, class_2):
    binary_mask = (labels == class_1) | (labels == class_2)
    binary_images = images[binary_mask]
    binary_labels = labels[binary_mask]
    return binary_images, binary_labels
