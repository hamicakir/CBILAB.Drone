import cv2
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
def predict(test_img):
    img=test_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    if gray is None:
        return None
    faces = faceCascade.detectMultiScale(gray, 1.3, 8)
    if len(faces) == 0:
        return None, None
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face,rect=gray[y:y + w, x:x + h],(x,y,w,h)
        label,confidence=recognizer.predict(face)
        label_text = subjects[label]
        draw_text(img, label_text, rect[0], rect[1] - 5)
        draw_text(img, str(confidence), rect[0]-100, rect[1]+200)
    return img
recognizer.read('Trainers/poi.yml')
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x+5, y+5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 6)
test=cv2.imread("poi1.jpg")
test2=cv2.imread("poi2.jpg")
test3=cv2.imread("poi3.jpg")
test4=cv2.imread("poi4.jpg")
test5=cv2.imread("poi5.jpg")
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
