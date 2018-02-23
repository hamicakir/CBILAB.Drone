import cv2
import os
import numpy as np
labels=[]
faces=[]
subjects = ["", "Admin", "Reese","Root","Fusco","Shaw"]
recognizer=cv2.face.LBPHFaceRecognizer_create()
def detect_face(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    if gray is None:
        return None
    faces=faceCascade.detectMultiScale(gray,1.2,10)
    if len(faces)==0:
        return None,None
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0), 2)
    return gray[y:y+w,x:x+h],faces[0]
path1='C:/Users/EMIRCAN/PycharmProjects/OPENCV1'
dirs=os.listdir(path1)
for dir in dirs:
    if not dir.startswith('s'):
        continue
    label=int(dir.replace('s',''))
    for image_name in os.listdir(path1+ "/" + dir):
        if image_name.startswith("."):
            continue
        image_path=path1+'/'+dir+'/'+image_name
        image=cv2.imread(image_path)
        if image is None:
            break
        face,rect=detect_face(image)
        cv2.imshow('Training on image',image)
        cv2.waitKey(1)
        if face is not None:
            faces.append(face)
            labels.append(label)
recognizer.train(faces,np.array(labels))
recognizer.save('Trainers/poi.yml')
print "Yuz sayisi:"+str(len(faces))
print "Etiket sayisi: "+str(len(labels))
print "You are being watched"
