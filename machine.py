import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D  # Import Conv2D
from tensorflow.keras import models, layers, datasets, utils
import matplotlib.pyplot as plt

# Global params for loading the images
IMG_ROOT_DIR = "C:/Faster_Code/ASSN_7_INT_SYSTEMS/QR_codes"
batch_size = 8
img_height = 180
img_width = 180

data = utils.image_dataset_from_directory(
    IMG_ROOT_DIR,
    image_size=(img_height, img_width),
    batch_size=batch_size)

qr_train, qr_test = utils.split_dataset(
    data, 
    left_size=0.7,
    shuffle = True,
    seed = 123)

qr_train, qr_val = utils.split_dataset(
    qr_train, 
    left_size=0.8,
    shuffle = True,
    seed = 123)

# Preview images
plt.figure(figsize=(10, 10))
for images, labels in qr_train.take(1):
  for i in range(8):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(qr_train.class_names[labels[i]])
    plt.axis("off")

# Show image sizes and other info (batch_size, height, width, color_channels)
for image_batch, labels_batch in qr_train:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Normalization / Scaling of colors
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = qr_train.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Do the same for the validation set
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = qr_val.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

# Do the same for the testing set
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = qr_test.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

# Cache datasets for performance
AUTOTUNE = tf.data.AUTOTUNE

qr_train = qr_train.cache().prefetch(buffer_size=AUTOTUNE)
qr_val = qr_val.cache().prefetch(buffer_size=AUTOTUNE)
qr_test = qr_test.cache().prefetch(buffer_size=AUTOTUNE)

# Create a model
num_classes = 2

model = tf.keras.Sequential([
  #tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(img_width, img_height, 3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

# Compile (Adam)
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.summary()
'''
model.fit(
  qr_train,
  validation_data=qr_val,
  epochs=3
)

evaluation = model.evaluate(qr_test)
print("Accuracy of testing: {}".format(evaluation[1]))
'''