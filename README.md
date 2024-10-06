
# Dokumentasi Model Pengenalan Gender Menggunakan TensorFlow dan Keras

Proyek ini menggunakan TensorFlow dan Keras untuk membangun dan melatih model Convolutional Neural Network (CNN) untuk pengenalan gender. Model ini dilatih pada dataset **CelebA**, yang terdiri dari 200.000 gambar, dengan tujuan untuk mengklasifikasikan gambar sebagai laki-laki atau perempuan.

## Daftar Isi
1. [Pengaturan Proyek](#setup)
2. [Pra-pemrosesan Data](#data-preprocessing)
3. [Arsitektur Model](#model-architecture)
4. [Kompilasi dan Pelatihan Model](#training)
5. [Evaluasi Model](#evaluation)
6. [Inferensi Model](#inference)

---

<a name="setup"></a>
## 1. Pengaturan Proyek

### a. Impor Dependensi

Berikut ini adalah pustaka yang digunakan dalam proyek:
- **TensorFlow/Keras** untuk membangun model CNN.
- **ImageDataGenerator** untuk pra-pemrosesan dan augmentasi gambar.
- **Google Colab** untuk integrasi drive dan menyimpan model.

```python
import os
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import zipfile
from google.colab import drive

drive.mount('/content/drive/')
```

### b. Ekstraksi Dataset

Dataset disimpan dalam file `.zip` di Google Drive dan diekstrak menggunakan modul `zipfile`.

```python
zip_ref = zipfile.ZipFile("/content/drive/MyDrive/DataSets/gender-recognition-200k-images-celeba.zip", 'r')
zip_ref.extractall("/tmp")
zip_ref.close()
```

---

<a name="data-preprocessing"></a>
## 2. Pra-pemrosesan Data

### a. Augmentasi dan Reskalasi Gambar

Untuk menghindari overfitting, teknik augmentasi data digunakan seperti rotasi, pergeseran, zooming, dan flipping.

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)
```

### b. Memuat Dataset

Dataset untuk pelatihan dan validasi dimuat dari direktori yang diekstrak. Gambar diubah ukurannya menjadi 64x64 piksel.

```python
train_generator = train_datagen.flow_from_directory(
    "/tmp/Dataset/Train",
    batch_size=256,
    class_mode='binary',
    target_size=(64, 64)
)

validation_generator = test_datagen.flow_from_directory(
    "/tmp/Dataset/Validation",
    batch_size=256,
    class_mode='binary',
    target_size=(64, 64)
)
```

---

<a name="model-architecture"></a>
## 3. Arsitektur Model

Model adalah arsitektur CNN yang terinspirasi dari AlexNet, dengan beberapa lapisan konvolusi yang diikuti oleh max-pooling, batch normalization, dan lapisan fully connected.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),

    tf.keras.layers.Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

---

<a name="training"></a>
## 4. Kompilasi dan Pelatihan Model

Model dikompilasi menggunakan **Adam optimizer** dan **binary crossentropy** sebagai fungsi loss karena ini adalah tugas klasifikasi biner. Akurasi digunakan sebagai metrik evaluasi.

```python
from keras.optimizers import Adam
model.compile(
    optimizer=Adam(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Pelatihan Model

Model dilatih selama 5 epoch dengan ukuran batch 256.

```python
hist = model.fit_generator(
    generator=train_generator,
    validation_data=validation_generator,
    steps_per_epoch=256,
    validation_steps=256,
    epochs=5
)
```

---

<a name="evaluation"></a>
## 5. Evaluasi Model

Setelah model dilatih, dilakukan evaluasi menggunakan dataset validasi.

```python
scores = model.evaluate(validation_generator, steps=89)
print("Akurasi: %.2f%%" % (scores[1]*100))
```

---

<a name="inference"></a>
## 6. Inferensi Model

Untuk melakukan inferensi pada model, gambar tunggal dapat diunggah dan diklasifikasikan sebagai laki-laki atau perempuan.

```python
uploaded = files.upload()

for fn in uploaded.keys():
   path = fn
   img = image.load_img(path, target_size=(64, 64))
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   images = np.vstack([x])
   classes = model.predict(images, batch_size=1)
   if classes[0] > 0.5:
      print("is a man")
   else:
      print("is a female")
```

---

## Kesimpulan
Proyek ini menunjukkan bagaimana membangun, melatih, dan mengevaluasi CNN untuk pengenalan gender menggunakan TensorFlow dan Keras. Model ini dapat lebih dioptimalkan dengan menyetel hyperparameter atau dengan meningkatkan ukuran dataset.
