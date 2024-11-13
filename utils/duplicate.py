import os
from imagededup.methods import PHash

# Inisialisasi metode PHash untuk mendeteksi duplikat
phasher = PHash()

# Path ke folder dataset
dataset_path = 'path/to/your/dataset'

# Temukan duplikat dalam folder dataset
duplicates = phasher.find_duplicates(directory=dataset_path, min_similarity_threshold=0.9)

# Loop untuk menghapus duplikat
for original, duplicate_list in duplicates.items():
    if duplicate_list:  # Jika ada duplikat ditemukan
        for duplicate in duplicate_list:
            duplicate_path = os.path.join(dataset_path, duplicate)
            if os.path.exists(duplicate_path):
                print(f"Menghapus gambar duplikat: {duplicate_path}")
                os.remove(duplicate_path)

print("Proses penghapusan duplikat selesai.")
