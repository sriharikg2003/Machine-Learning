# imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


# using the saved model
model = tf.keras.models.load_model('handwritten.model')


actual=[]
predicted=[]
for i in range(10):
    print(i)
    actual.append(i)
    img = cv2.imread(f"digits/{i}.png")
    img = img[:,:,0]
    img = np.invert(np.array([img])) # we need black on white
    prediction = model.predict(img)
    print(f"Actual : {i}   ::  Predicted : {np.argmax(prediction)}")
    predicted.append(np.argmax(prediction))

print(f"Accuracy {(sum(np.array(actual)==np.array(predicted)))}")