import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Fungsi untuk memuat dan memproses gambar
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi
    return img_array

# Load model yang telah dilatih
model = tf.keras.models.load_model('model_penyakit_tanaman_kopi.h5')

# Masukkan path ke gambar yang ingin dideteksi
img_path = 'path/to/gambar_daun_kopi.jpg'
img = load_and_preprocess_image(img_path)

# Prediksi penyakit
predictions = model.predict(img)
class_names = ['Sehat', 'Karat Daun', 'Infeksi Jamur', 'Bercak Daun']  # Sesuai dengan label model Anda

# Menampilkan hasil prediksi
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print(f'Hasil Deteksi: {predicted_class} ({confidence:.2f}% yakin)')
