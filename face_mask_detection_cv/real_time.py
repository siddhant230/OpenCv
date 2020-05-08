import numpy as np
import cv2, argparse
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# load our serialized faceNet and MaskNet

prototxtPath = 'face_detector\\deploy.prototxt'
weightsPath = 'face_detector\\res10_300x300_ssd_iter_140000.caffemodel'
mask_path = 'face_detector\\mask_detector.model'

print("[INFO] loading faceNet...")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
print('faceNet loaded')

print("[INFO] loading MaskNet...")
maskNet = load_model(mask_path)
print('maskNet loaded')
print(maskNet.summary())

def detect_and_predict_mask(frame, faceNet, maskNet, confidence_thresh=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > confidence_thresh:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            try:
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
            except:
                pass

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        preds = maskNet.predict(faces)

    return (locs, preds)

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)

    (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > 0.005 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(img, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
