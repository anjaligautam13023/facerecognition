import cv2
def createdataset():
    face_cascade = cv2.CascadeClassifier('E:\Anjali\python\pythonProject\harcascade.xml')
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    count=0
    z=0
    windownname='test'
    while z<=z+1:

        _, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cropped_face = img[y:y + h, x:x + w]
            cv2.imwrite("E:/Anjali/python/pythonProject/dataset/1_shilpashetty/anjali{}.png".format(count),cropped_face )
            cv2.putText(img, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(windownname, img)
        cv2.setWindowProperty(windownname, cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) == 13 or count == 50:
                 break
        count=count+1
        z=z+1
    cap.release()
    cv2.destroyAllWindows()

