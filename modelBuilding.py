import cv2
import os
import numpy as np

data_path = 'data'


X_train = []
y_train = []
dict = {'hoan_1': 1, 'bao_2': 2, 'ngan_3': 3, 'chi_4': 4, 'hung_5': 5}  #encode 

for users in os.listdir(data_path):
    users_path = os.path.join(data_path,users)
    #print(users_path)  #data/hoan_1
    lst_user =[]
    for image in os.listdir(users_path):
        image_path = os.path.join(users_path,image)
        label = image_path.split('\\')[1]
        #print(image_path)  #data/hoan/anh
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lst_user.append(img) #them tung user vao list rieng, moi tam hinh kem 1 label 
        y_train.append(int(dict[label]))
    X_train.extend(lst_user) #them tat ca user vao list chung 


# print(X_train)
# print(y_train)
recognizer = cv2.face.LBPHFaceRecognizer_create()
print('\n [INFO] Training data...')
#faces, ids = getImagesAndLabels(path)
recognizer.train(X_train, np.array(y_train))
recognizer.write( 'trainer/trainer.yml')
print('Done')