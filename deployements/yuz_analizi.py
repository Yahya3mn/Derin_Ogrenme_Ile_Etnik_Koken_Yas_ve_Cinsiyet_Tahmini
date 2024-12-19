import cv2

def faceBox(faceNet,frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame,1.0,(227,227),[104,117,123],swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox = []
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.7:
            x1 = int(detection[0,0,i,3] * frameWidth)
            y1 = int(detection[0,0,i,4] * frameHeight)
            x2 = int(detection[0,0,i,5] * frameWidth)
            y2 = int(detection[0,0,i,6] * frameHeight)
            bbox.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
    return frame, bbox


faceProto = "deployements/opencv_face_detector.pbtxt"
faceModel = "deployements/opencv_face_detector_uint8.pb"

ageProto =  "deployements/age_deploy.prototxt"
ageModel =  "deployements/age_net.caffemodel"

genderProto =  "deployements/gender_deploy.prototxt"
genderModel =  "deployements/gender_net.caffemodel"

raceModel = "deployements/race_detection.keras"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

Model_Mean_Values = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
genderList = ['Male','Female']

video = cv2.VideoCapture(0)
padding = 20
while True:
    ret, frame = video.read()
    frame,bboxs = faceBox(faceNet,frame)
    for bbox in bboxs:
        #face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1), max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face,1.0,(227,227),Model_Mean_Values, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]


        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = "{},{}".format(gender,age)
        cv2.rectangle(frame, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1)
        cv2.putText(frame,label,(bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Age-Gender-Race",frame)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()