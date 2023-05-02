from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from minivggnet import MiniVGGNet  # Sửa dòng import này
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
import numpy as np
import argparse
import os


# Tải dữ liệu animals
def load_animals_data():
    # Tải dữ liệu animals gồm cat, dog, panda từ bộ dữ liệu CIFAR10.
    # Trong CIFAR10, các minh họa của cat, dog và panda tương ứng là 3, 5, 7.
    
    ((X_train, y_train), (X_test, y_test)) = cifar10.load_data()
    train_idxs =  np.where((y_train == 3) | (y_train == 5) | (y_train == 7))[0]
    test_idxs = np.where((y_test == 3) | (y_test == 5) | (y_test == 7))[0]
    
    X_train = X_train[train_idxs]
    y_train = y_train[train_idxs]
    X_test = X_test[test_idxs]
    y_test = y_test[test_idxs]

    return X_train, y_train, X_test, y_test

# Tiền xử lý dữ liệu
def preprocess_data(X_train, y_train, X_test, y_test):

    # Tăng cường dữ liệu
    dataGen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
                
    # Chuẩn hóa dữ liệu
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, X_test, dataGen

# Khởi tạo model
opt = SGD(lr=0.05, decay=0.0, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

(X_train, y_train, X_test, y_test) = load_animals_data()
(X_train, X_test, dataGen) = preprocess_data(X_train, y_train, X_test, y_test)

# Chuyển các nhãn thành dạng one-hot encoding
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Huấn luyện model
epochs = 60
batch_size = 64
H = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch_size), validation_data=(X_test, y_test), steps_per_epoch=len(X_train)//batch_size, epochs=epochs, verbose=1)

# Đánh giá model
predictions = model.predict(X_test, batch_size=64)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))

# Lưu model dưới dạng H5
model.save("trained_model.h5")

# Vẽ biểu đồ
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("train_val_loss_acc_plot.png")
