import cv2 
import time
import os
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
count = 1
face_id = input('\n Input ID of user ==>  ')
name = input('\n Input your name:')

print("\n [INFORMATION] Create Camera...")
while True:
    _, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img_gray, 1.3, 5)
    #time.sleep(0.5)
    for (x,y,w,h) in faces:  #x la hoanh do(cot), y la tung do(dong), w va h la chieu dai, chieu rong
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        #img_face = img_gray[y+2: y+h-2, x+2 :x+w-2]
        img_face = cv2.resize(img_gray[y+2: y+h-2, x+2 :x+w-2], [100,100]) #cắt ảnh từng khuôn mặt

        if not os.path.isdir('data/' + str(name) + '_' + str(face_id)):
            os.mkdir('data/' + str(name) + '_' + str(face_id))
            #print('data/' + str(name)+ '_' + str(face_id))
            
        cv2.imwrite('data/'+ str(name) + '_' + str(face_id) + '/anh_{}.jpg'.format(count), img_face)
        count += 1

    cv2.imshow("frame" , img)
    k = cv2.waitKey(100) & 0xff
    if k == 27: #esc
        break
    elif count == 51:
        break

print("\n [INFORMATION] Exit")
cap.release()
cv2.destroyAllWindows()