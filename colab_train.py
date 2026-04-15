from google.colab import drive
drive.mount('/content/drive')

# imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os, cv2, random

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

from sklearn.metrics import confusion_matrix, classification_report

# speed improvement
mixed_precision.set_global_policy('mixed_float16')

# copying dataset from drive
!cp -r "/content/drive/MyDrive/Colab Notebooks/Training" /content/
!cp -r "/content/drive/MyDrive/Colab Notebooks/Testing" /content/

train_path = "/content/Training"
test_path = "/content/Testing"


# basic settings
IMG_SIZE = 112
BATCH_SIZE = 64

# loading dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

# simple augmentation
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (data_aug(preprocess_input(x)), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

train_ds = train_ds.shuffle(500).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# model (transfer learning)
base_model = MobileNetV2(weights='imagenet', include_top=False,
                         input_shape=(IMG_SIZE, IMG_SIZE, 3))

# freezing most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# last few layers trainable
for layer in base_model.layers[-30:]:
    layer.trainable = True


x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

output = Dense(num_classes, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6
)


# training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop, lr_reduce]
)


# saving model
model.save("brain_tumor_model_fast.h5")


# plotting results
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["train", "val"])

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["train", "val"])

plt.show()

# confusion matrix
y_pred, y_true = [], []

for imgs, labels in test_ds:
    preds = model.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))

# single image prediction
def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        print("image not found")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    idx = np.argmax(pred)

    print("Prediction:", class_names[idx])
    print("Confidence:", pred[0])


# multiple images
def predict_multiple(folder, n=5):
    imgs = os.listdir(folder)
    imgs = random.sample(imgs, min(n, len(imgs)))

    plt.figure(figsize=(15,6))

    for i, name in enumerate(imgs):
        path = os.path.join(folder, name)
        img = cv2.imread(path)

        if img is None:
            continue

        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        inp = preprocess_input(resized)
        inp = np.expand_dims(inp, axis=0)

        pred = model.predict(inp, verbose=0)
        idx = np.argmax(pred)

        plt.subplot(1, len(imgs), i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(class_names[idx])
        plt.axis('off')

    plt.show()



# ✅ USAGE
# predict_image("/content/Testing/some_image.jpg")
predict_multiple("/content/drive/MyDrive/Colab Notebooks/Testing/glioma_tumor", 5)