import cv2
import os
import numpy as np
labels=[]
faces=[]
subjects = ["", "Admin", "Reese","Root","Fusco","Shaw"]
recognizer=cv2.face.LBPHFaceRecognizer_create()
def detect_face(img):
    grays,rects=[],[]
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
def predict(test_img):
    img=test_img.copy()
    face,rect=detect_face(img)
    label,confidence=recognizer.predict(face)
    label_text = subjects[label]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1] - 5)
    draw_text(img, str(confidence), rect[0]-100, rect[1]+200)
    return img
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
print len(faces)
print len(labels)
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x+5, y+5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 6)
test=cv2.imread("finch.jpg")
test2=cv2.imread("reese.jpg")
test3=cv2.imread("root.jpg")
test4=cv2.imread("fusco.jpg")
test5=cv2.imread("poi.jpg")
predicted_img1=predict(test)
predicted_img2=predict(test2)
predicted_img3=predict(test3)
predicted_img4=predict(test4)
predicted_img5=predict(test5)
cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.imshow(subjects[3], cv2.resize(predicted_img3, (400, 500)))
cv2.imshow(subjects[4], cv2.resize(predicted_img4, (400, 500)))
cv2.imshow(subjects[5], cv2.resize(predicted_img5, (400, 500)))
cv2.waitKey(100000)
cv2.destroyAllWindows()
