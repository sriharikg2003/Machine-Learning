# imports
import numpy as np
print("Imported numpy successfully")
import matplotlib.pyplot as plt
import tensorflow as tf
print("Imported tensorflow successfully")
import cv2
print("Imported cv2 successfully")


# using the saved model
model = tf.keras.models.load_model('handwritten.model')
print("Loaded model successfully")

print("Opening Camera")
cap = cv2.VideoCapture(1)

print("\n*************************\nPrediction Starts")

while True:

    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',img)
    img_resized = cv2.resize(gray, (28, 28))
    img_resized = np.invert(np.array([img_resized]))
    prediction = model.predict(img_resized)
    print("\n**************************\nDIGIT RECOGNIZE IS ", np.argmax(prediction))
 

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
cap.release()