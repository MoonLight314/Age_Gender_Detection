import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model, save_model

MODEL_FILE = "opencv_face_detector_uint8.pb"
CONFIG_FILE = "opencv_face_detector.pbtxt"
SIZE = 300
CONFIDENCE_FACE = 0.9

AGE_BAND_RESULT = ["20~30","30~40","40~50","Over 50"]
GENDER_RESULT = ["Man","Woman"]

print(tf.__version__)

# Load Face Detection Model
net = cv2.dnn.readNetFromTensorflow( MODEL_FILE , CONFIG_FILE )

print("Loading Model...")

model = load_model("CheckPoints_K-Face_Gender_Age_Band_F1_Score")

cap = cv2.VideoCapture(0)

while cv2.waitKey(1) < 0:
    ret, frame = cap.read()
    rows, cols, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1.0, (SIZE, SIZE))  # , (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for detection in detections[0, 0]:

        score = float(detection[2])

        if score > CONFIDENCE_FACE:

            if detection[3] >= 1.00 or detection[4] >= 1.00 or detection[5] >= 1.00 or detection[6] >= 1.00 or detection[3] <= 0 or detection[4] < 0 or detection[5] <= 0 or detection[6] <= 0:
                pass
            else:
                left = int(detection[3] * cols)
                top = int(detection[4] * rows)
                right = int(detection[5] * cols)
                bottom = int(detection[6] * rows)

                cropped = frame[top:bottom, left:right]
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                cropped = cv2.resize( cropped , dsize=(224,224) )
                cropped = np.array(cropped).reshape(-1,224,224,3)

                pred = model.predict( cropped )
                
                gender_result = pred[0]
                age_band_result = pred[1]

                AGE_BAND_RESULT = ["20~30","30~40","40~50","Over 50"]
                GENDER_RESULT = ["Man","Woman"]
                
                Result = "{0} , {1}".format(GENDER_RESULT[int(np.argmax(np.reshape( gender_result , (1,-1) )))] , AGE_BAND_RESULT[int(np.argmax(np.reshape( age_band_result , (1,-1) )))])

                cv2.putText(frame, Result, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)


    cv2.imshow("VideoFrame", frame)

cap.release()
cv2.destroyAllWindows()
