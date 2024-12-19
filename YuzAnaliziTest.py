import cv2
from tensorflow.keras.models import load_model
import numpy as np

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bbox.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bbox

# Model yükleme
faceProto = "deployements/opencv_face_detector.pbtxt"
faceModel = "deployements/opencv_face_detector_uint8.pb"

genderProto = "deployements/gender_deploy.prototxt"
genderModel = "deployements/gender_net.caffemodel"

ageProto = "deployements/age_deploy.prototxt"
ageModel = "deployements/age_net.caffemodel"

raceModelPath = "deployements/ethnicity_model.keras"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
raceNet = load_model(raceModelPath) 

Model_Mean_Values = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
raceList = ['White', 'Black', 'Asian', 'Indian', 'Others']

video = cv2.VideoCapture(0)
padding = 20

while True:
    ret, frame = video.read()
    frame, bboxs = faceBox(faceNet, frame)
    for bbox in bboxs:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), Model_Mean_Values, swapRB=False)

        # Gender Prediction
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Age Prediction
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Race Prediction
        face_resized = cv2.resize(face, (48, 48))  
        face_resized_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)  
        face_array = np.expand_dims(face_resized_gray, axis=(0, -1)) / 255.0  

        racePreds = raceNet.predict(face_array)
        race = raceList[racePreds[0].argmax()]  

        # Label oluşturma
        label = "{}, {}, {}".format(gender, age, race)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 128, 0), -1)  

        font_scale = 1.0
        font_thickness = 2
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    cv2.imshow("Gender-Age-Race Detection", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
