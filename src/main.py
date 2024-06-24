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
    # plota a matriz de confusão como um mapa de calor e salva dentro de /plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(filename)
    plt.close()

# carrega e pré-processa os dados
(train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

# constroi e treina o modelo multiclasses
print("\n \033[95m Training and evaluating multiclass model... \033[0m")
multiclass_model = build_multiclass_model((28, 28, 1))
_, _, _, multiclass_predictions = train_and_evaluate(multiclass_model, train_images, train_labels, test_images, test_labels)

# transforma as previsões do modelo multiclasses para rótulos de classes
multiclass_pred_labels = np.argmax(multiclass_predictions, axis=1)

# calcula a matriz de confusão para o modelo multiclasses
multiclass_cm = confusion_matrix(test_labels, multiclass_pred_labels)
plot_confusion_matrix(multiclass_cm, class_names=[str(i) for i in range(10)], title='Multiclass Confusion Matrix', filename='plot/multiclass_confusion_matrix.png')

# constroi e treina o modelo binário
target_class = 0
print(f"\n \033[95m Training and evaluating binary model for class {target_class}... \033[0m")
binary_train_images, binary_train_labels = filter_binary_classes(train_images, train_labels, target_class)
binary_test_images, binary_test_labels = filter_binary_classes(test_images, test_labels, target_class)
binary_model = build_binary_model((28, 28, 1))
_, _, _, binary_predictions = train_and_evaluate(binary_model, binary_train_images, binary_train_labels, binary_test_images, binary_test_labels)

# transforma as previsões do modelo binário para rótulos de classes
binary_pred_labels = (binary_predictions > 0.5).astype(int).flatten()

# calcula a matriz de confusão para o modelo binário
binary_cm = confusion_matrix(binary_test_labels, binary_pred_labels)
plot_confusion_matrix(binary_cm, class_names=['Not ' + str(target_class), str(target_class)], title=f'Binary Confusion Matrix (Not {target_class} vs {target_class})', filename=f'plot/binary_confusion_matrix_{target_class}.png')