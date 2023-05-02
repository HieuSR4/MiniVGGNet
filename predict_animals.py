#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from tensorflow import keras
#from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os

dir_path = "image"

def preprocess_image(image, width, height):
    image = cv2.resize(image, (width, height))
    image = image.astype("float")/255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(model, image_path):
    image = cv2.imread(image_path)
    output = imutils.resize(image, width=400)

    preprocessed_image = preprocess_image(image, 32, 32)
    proba = model.predict(preprocessed_image)[0]

    return output, proba

model = keras.models.load_model('D:/trainedmodel.h5')

for img in os.listdir(dir_path):
    image_path = os.path.join(dir_path, img)
    original_image, proba = predict_image(model, image_path)

    label = ["cat", "dog", "panda"][np.argmax(proba)]
    text = "{}: {:.2f}%".format(label, proba[np.argmax(proba)] * 100)
    cv2.putText(original_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Image", original_image)
    cv2.waitKey(0)

