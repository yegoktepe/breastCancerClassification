import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import xgboost as xgb

# Veri yükleme ve önişleme
data = pd.read_csv('data.csv')
X = data.drop(columns=[data.columns[1]])  # İkinci sütun sınıf bilgisini içeriyor, onu çıkarıyoruz
y = data[data.columns[1]]  # İkinci sütun sınıf hedefidir

# Yeni özellikler oluşturma
X['radius_texture_diff'] = X.iloc[:, 0] - X.iloc[:, 1]  # İlk iki sütunun farkı
X['feature_std_dev'] = X.std(axis=1)  # Tüm özelliklerin standart sapması
X['feature_variance'] = X.var(axis=1)  # Tüm özelliklerin varyansı

# Sınıf etiketlerini sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Veriyi normalize etme
scaler = StandardScaler()
X = scaler.fit_transform(X)

# PCA ile boyut indirgeme
pca = PCA(n_components=20)  # Burada bileşen sayısını ayarlayabilirsiniz
X_pca = pca.fit_transform(X)

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = X_train.reshape(-1, X_train.shape[1], 1)  # CNN için yeniden şekillendirme
X_test = X_test.reshape(-1, X_test.shape[1], 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# İlk Katman: CNN Modelleri
def create_cnn():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 1), activation='relu', input_shape=(X_train.shape[1], 1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

# İlk Katman: Dense Model
def create_dense():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

# İlk katmandaki CNN ve Dense modellerini eğitme
cnn_models = [create_cnn() for _ in range(3)]
dense_models = [create_dense() for _ in range(2)]

for model in cnn_models + dense_models:
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.3, callbacks=[EarlyStopping(patience=3)], verbose=1)

# İlk katmandaki tüm modellerin tahminlerini elde etme
cnn_predictions = [model.predict(X_train) for model in cnn_models]
dense_predictions = [model.predict(X_train) for model in dense_models]

# İkinci katman için giriş verisi olarak ilk katmandan gelen tahminlerin birleştirilmesi
stacked_train_input = np.concatenate(cnn_predictions + dense_predictions, axis=1)
stacked_train_input = stacked_train_input.reshape(stacked_train_input.shape[0], -1)  # XGBoost uyumlu hale getirme

# İkinci Katman: XGBoost Modeli ile Meta-Öğrenme
xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, random_state=42)
y_train_binary = np.argmax(y_train, axis=1)  # y_train'i XGBoost için 0 ve 1 sınıflarına dönüştürme
xgb_model.fit(stacked_train_input, y_train_binary)

# Test verisi için ilk katmandaki tüm modellerin tahminlerini elde etme
cnn_test_predictions = [model.predict(X_test) for model in cnn_models]
dense_test_predictions = [model.predict(X_test) for model in dense_models]
stacked_test_input = np.concatenate(cnn_test_predictions + dense_test_predictions, axis=1)
stacked_test_input = stacked_test_input.reshape(stacked_test_input.shape[0], -1)  # XGBoost uyumlu hale getirme

# XGBoost modeli ile test verisinde tahmin yapma
y_pred = xgb_model.predict(stacked_test_input)

# Sonuçları yazdırma
y_test_binary = np.argmax(y_test, axis=1)  # y_test'i XGBoost için 0 ve 1 sınıflarına dönüştürme
print("Accuracy:", accuracy_score(y_test_binary, y_pred))
print("Classification Report:\n", classification_report(y_test_binary, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_binary, y_pred))

# Eğitim kümesi için değerlendirme
y_train_pred = xgb_model.predict(stacked_train_input)
print("Training Set Accuracy:", accuracy_score(y_train_binary, y_train_pred))
print("Training Set Classification Report:\n", classification_report(y_train_binary, y_train_pred))
print("Training Set Confusion Matrix:\n", confusion_matrix(y_train_binary, y_train_pred))



# tüm veri seti için değerlendirme
# Eğitim ve test verilerini birleştirme
X_full = np.concatenate([X_train, X_test], axis=0)
y_full = np.concatenate([y_train, y_test], axis=0)

# İlk katmandaki CNN ve Dense modellerinden tüm veri kümesi üzerinde tahminler elde etme
cnn_full_predictions = [model.predict(X_full) for model in cnn_models]
dense_full_predictions = [model.predict(X_full) for model in dense_models]
stacked_full_input = np.concatenate(cnn_full_predictions + dense_full_predictions, axis=1)
stacked_full_input = stacked_full_input.reshape(stacked_full_input.shape[0], -1)  # XGBoost uyumlu hale getirme

# XGBoost modeli ile tüm veri kümesinde tahmin yapma
y_full_pred = xgb_model.predict(stacked_full_input)

# Tüm veri kümesi için orijinal sınıf etiketlerine dönüştürme
y_full_binary = np.argmax(y_full, axis=1)

# Tüm veri kümesi için sonuçları yazdırma
print("Full Dataset Accuracy:", accuracy_score(y_full_binary, y_full_pred))
print("Full Dataset Classification Report:\n", classification_report(y_full_binary, y_full_pred))
print("Full Dataset Confusion Matrix:\n", confusion_matrix(y_full_binary, y_full_pred))


# Öznitelik uzayı görselleştirlmesi
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# Function for 3D feature space visualization with legend
def visualize_feature_space_3d(X, y, title="Feature Space", use_pca=False):
    if use_pca:
        reducer = PCA(n_components=3)
    else:
        reducer = TSNE(n_components=3, random_state=2)

    X_reduced = reducer.fit_transform(X)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
                          c=y, cmap="viridis", alpha=0.7, label=y)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")

    # Legend için sınıf isimleri
    legend_labels = {0: 'Benign', 1: 'Malignant'}
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[label],
                          markersize=10, markerfacecolor=plt.cm.viridis(i / 1.0))
               for i, label in enumerate(legend_labels.keys())]
    ax.legend(handles=handles, title="Classes")
    
    plt.show()

# Visualizing 3D feature space without PCA (using all original features)
visualize_feature_space_3d(X, y, title="3D Feature Space without PCA", use_pca=False)

# Visualizing 3D feature space with PCA (using first 20 components)
visualize_feature_space_3d(X_pca, y, title="3D Feature Space with PCA", use_pca=True)






# karmaşıklık matrisi, roc ve pr eğrilerinin çizimi
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve

# Confusion Matrix çizme fonksiyonu
def plot_confusion_matrix(y_true, y_pred, title="UNNEBC Model"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

# ROC ve PR eğrilerini çizme fonksiyonu
def plot_roc_pr_curves(y_true, y_probs):
    # ROC eğrisi
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    
    # PR eğrisi
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
    pr_auc = auc(recall, precision)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.show()

# Confusion matrix görselleştirme
plot_confusion_matrix(y_full_binary, y_full_pred)

# ROC ve PR eğrilerini çizme
# XGBoost modelinden sınıf olasılıklarını tahmin etme
y_probs = xgb_model.predict_proba(stacked_test_input)  # Sınıf olasılıklarını elde etme
plot_roc_pr_curves(y_test_binary, y_probs)


