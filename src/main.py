import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_and_preprocess_data, filter_binary_classes
from src.cnn_model import build_multiclass_model, build_binary_model
from src.train_and_evaluate import train_and_evaluate

# Carregar e preprocessar dados
(train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

# Tarefa multiclasse
print("Training and evaluating multiclass model...")
multiclass_model = build_multiclass_model((28, 28, 1))
train_and_evaluate(multiclass_model, train_images, train_labels, test_images, test_labels)

# Tarefa de classificação binária (ex. dígitos 0 e 1)
class_1, class_2 = 0, 1
print(f"\nTraining and evaluating binary model for classes {class_1} and {class_2}...")
binary_train_images, binary_train_labels = filter_binary_classes(train_images, train_labels, class_1, class_2)
binary_test_images, binary_test_labels = filter_binary_classes(test_images, test_labels, class_1, class_2)
binary_model = build_binary_model((28, 28, 1))
train_and_evaluate(binary_model, binary_train_images, binary_train_labels, binary_test_images, binary_test_labels)
