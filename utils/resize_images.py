import os
import cv2
from PIL import Image

# Path folder input dan output
input_folder = '/home/pandu/ya/data/deteksiDaun'
output_folder = '/home/pandu/ya/data/deteksi_resize'

# Ukuran output yang diinginkan
desired_width, desired_height = 256, 256

# Pastikan folder output ada
os.makedirs(output_folder, exist_ok=True)

# Fungsi untuk mendeteksi kontur dan memusatkan subjek
def center_object(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Deteksi kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Dapatkan kotak pembatas terbesar
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop gambar berdasarkan kotak pembatas
        img_cropped = img[y:y+h, x:x+w]

        return img_cropped
    return img

# Proses semua file dalam subfolder
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter gambar
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)

            if img is not None:
                # Deteksi dan pusatkan subjek
                img_centered = center_object(img)

                # Ubah ke format PIL untuk resize
                img_pil = Image.fromarray(cv2.cvtColor(img_centered, cv2.COLOR_BGR2RGB))
                img_resized = img_pil.resize((desired_width, desired_height))

                # Buat path output dengan struktur yang sama
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)

                # Simpan hasil di folder output
                output_path = os.path.join(output_subfolder, filename)
                img_resized.save(output_path)

print("Proses deteksi kontur dan resize selesai.")
