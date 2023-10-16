import cv2
import numpy as np
import os

# Data
dataset_path = "data/"
face_data = []
labels = []
name_map = {}

class_id=0

for f in os.listdir(dataset_path):
    if f.endswith('.npy'):

        # Mapping
        # remove the last 4 characters(.npy)
        name_map[class_id]=f[:-4]

        # X-values
        data_item=np.load(dataset_path+f)
        m=data_item.shape[0]
        face_data.append(data_item)

        # Y-values
        target=class_id*np.ones((m))
        class_id+=1
        labels.append(target)

# print(face_data);
# print(labels);

XT=np.concatenate(face_data,axis=0)
yT=np.concatenate(labels,axis=0).reshape((-1))

print(XT.shape)
print(yT.shape)
print(name_map)


# Algorithm

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,y,xt,k=5):

    m=X.shape[0]
    dlist=[]

    for i in range(m):
        d=dist(X[i],xt)
        dlist.append([d,y[i]])
    
    dlist=sorted(dlist)
    # print(dlist)
    dlist=np.array(dlist[:k])
    labels=dlist[:,1]

    labels,cnts=np.unique(labels,return_counts=True)
    idx=np.argmax(cnts)
    pred=labels[idx]

    return int(pred)

# Predictions

# 0: default camera
cam = cv2.VideoCapture(0)

# Model
model=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml');

# read image from camera object
while True:
    success, img = cam.read()

    if not success:
        print("Failed to read image from camera")
        exit(-1)

    faces = model.detectMultiScale(img, scaleFactor=1.05,minNeighbors=5)

    # render the box around each face and predict the name
    for f in faces:
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # crop the face largest face
        # offset=10
        # if y-offset>0 and x-offset>0:
        #     cropped_face = img[y-offset:y+h+offset, x-offset:x+w+offset]
        # else:
        cropped_face = img[y:y+h, x:x+w]
        
        cropped_face = cv2.resize(cropped_face, (100, 100))

        # predict the name
        pred=knn(XT,yT,cropped_face.flatten())

        # map the id to name
        pred_name=name_map[pred]

        cv2.putText(img,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.imshow("Cropped Face", cropped_face)
    cv2.imshow("Prediction Window", img)
    key = cv2.waitKey(1)  # 1ms pause for every frame to show next image
    if key == ord('q'):
        break

# release camera, close window
cam.release()
cam.destroyAllWindows()
