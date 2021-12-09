import cv2
from os import listdir
import os
import numpy as np
from os.path import isfile, join

def detect_face():

    data_path='E:/Anjali/python/pythonProject/dataset/'
    #onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]
   # print(onlyfiles)
    dirs = os.listdir(data_path)
    #print(dirs)
    dataset = []
    arr=[]

    WindowName="mainview"

    training_data,lables=[],[]
    for dir_name in dirs:
        subject_dir_path = data_path  + dir_name
        dataset.append(dir_name.split('_')[-1])
        #print(subject_dir_path)
        subject_images_names = os.listdir(subject_dir_path)

        # print(subject_images_names[0].split('_')[-1])
       # dataset.append( subject_images_names.split("_")[1])
       # print(dataset)
        for image in subject_images_names:
            image_path = subject_dir_path +"/"+ image
            #print(image_path)
            name = image_path.split('/')[-2]
            #print(name)

            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            training_data.append(np.asarray(images, dtype=np.uint8))
            lables.append(name.split("_")[0])
            #print(training_data)
   # print(dataset)

   # print(lables)
    lables = np.asarray(lables,dtype=np.int32)
    #print(lables)
    model=cv2.face.LBPHFaceRecognizer_create()
    #print(type(model))
    model.train(np.asarray(training_data),np.asarray(lables))
    face_classifier=cv2.CascadeClassifier('E:\Anjali\python\pythonProject\harcascade.xml')
    def face_extractor(img,):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces is():
            return img,[]
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped_face=img[y:y+h,x:x+w]

        return img,cropped_face
    cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count=0
    z=0
    while z<=150:
        ret ,frame=cap.read()
        image,face=face_extractor(frame)
        try:
           #print(face)
           face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
           result=model.predict(face)
          # print(result)

           if result[1]<500:
               confidence=int(100*(1-(result[1])/300))
               #print(confidence)
           if confidence>82:
               cv2.putText(image,"HLO "+dataset[result[0]-1]+" you are allowed",(100,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
               cv2.imshow(WindowName, image)
               cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
               arr.append(dataset[result[0]-1])
           else:
               cv2.putText(image, "you are not allowed", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
               cv2.imshow(WindowName, image)
               cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
               arr.append(0)

        except:
              cv2.putText(image, "face not recognised", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
              cv2.imshow(WindowName,image)
              cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)
        z = z + 1
        if cv2.waitKey(1)==13 or cv2.waitKey(1)==32 :
            cap.release()
            break

    r=max(arr, key=arr.count)
    cap.release()
    cv2.destroyAllWindows()
    return r



