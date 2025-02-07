import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load kembali model yang telah disimpan
model = load_model("model_rempah.h5")

# Direktori dataset uji (pastikan sesuai dengan struktur folder kamu)
test_dir = "dataset/test"

# Preprocessing untuk data uji
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),  # Ukuran gambar sesuai model
    batch_size=32,
    class_mode="categorical",
    shuffle=False  # Harus False agar label sesuai dengan prediksi
)

# Mapping label ke indeks
labels = list(test_generator.class_indices.keys())


# Ambil data uji (X_test) dan label aslinya (y_test)
X_test, y_test = next(test_generator)

# Lakukan prediksi
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Ambil kelas dengan probabilitas tertinggi
y_true = np.argmax(y_test, axis=1)  # Ambil label sebenarnya

# Hitung Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Buat plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Jahe', 'Kencur', 'Kunyit', 'Lengkuas'],
            yticklabels=['Jahe', 'Kencur', 'Kunyit', 'Lengkuas'])

plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix")
plt.show()

# Print laporan klasifikasi untuk melihat metrik Precision, Recall, dan F1-score
print("Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=['Jahe', 'Kencur', 'Kunyit', 'Lengkuas']))

# Misal, cm adalah confusion matrix yang sudah dihitung
correct_predictions = np.trace(cm)  # Jumlah elemen diagonal utama
total_predictions = np.sum(cm)
accuracy_percentage = (correct_predictions / total_predictions) * 100

print("Akurasi (dihitung dari confusion matrix): {:.2f}%".format(accuracy_percentage))