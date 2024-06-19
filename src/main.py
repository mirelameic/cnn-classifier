import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_and_preprocess_data, filter_binary_classes
from src.cnn_model import build_multiclass_model, build_binary_model
from src.train_and_evaluate import train_and_evaluate

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', filename='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(filename)
    plt.close()

# carregar e pré-processar os dados
(train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

# construir e treinar o modelo multiclass
print("Training and evaluating multiclass model...")
multiclass_model = build_multiclass_model((28, 28, 1))
_, _, _, multiclass_predictions = train_and_evaluate(multiclass_model, train_images, train_labels, test_images, test_labels)

# transformar as previsões do modelo multiclass para rótulos de classes
multiclass_pred_labels = np.argmax(multiclass_predictions, axis=1)

# calcular a matriz de confusão para o modelo multiclass
multiclass_cm = confusion_matrix(test_labels, multiclass_pred_labels)
plot_confusion_matrix(multiclass_cm, class_names=[str(i) for i in range(10)], title='Multiclass Confusion Matrix', filename='plot/multiclass_confusion_matrix.png')

# construir e treinar o modelo binário
class_1, class_2 = 0, 1
print(f"\nTraining and evaluating binary model for classes {class_1} and {class_2}...")
binary_train_images, binary_train_labels = filter_binary_classes(train_images, train_labels, class_1, class_2)
binary_test_images, binary_test_labels = filter_binary_classes(test_images, test_labels, class_1, class_2)
binary_model = build_binary_model((28, 28, 1))
_, _, _, binary_predictions = train_and_evaluate(binary_model, binary_train_images, binary_train_labels, binary_test_images, binary_test_labels)

# transformar as previsões do modelo binário para rótulos de classes
binary_pred_labels = (binary_predictions > 0.5).astype(int).flatten()

# calcular a matriz de confusão para o modelo binário
binary_cm = confusion_matrix(binary_test_labels, binary_pred_labels)
plot_confusion_matrix(binary_cm, class_names=[str(class_1), str(class_2)], title=f'Binary Confusion Matrix ({class_1} vs {class_2})', filename='plot/binary_confusion_matrix.png')
