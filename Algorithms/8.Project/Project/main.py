import cv2
import numpy as np

# 0: default camera
cam = cv2.VideoCapture(0)

# Ask the name
name = input("Enter your name: ")

# Create a folder with the name
dataset_path = "data/"

offset=20

# create a list of face data
face_data=[]
cnt=0;
skip=0

# Model
model=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml');

# read image from camera object
while True:
    success, img = cam.read()

    if not success:
        print("Failed to read image from camera")
        exit(-1)

    # store the gray image
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(img, scaleFactor=1.05,minNeighbors=5)

    # pick the largest face in the bounding box
    faces=sorted(faces,key=lambda f:f[2]*f[3])

    # pick the last face (because it is the largest face acc to area(f[2]*f[3]))
    if(len(faces)!=0):
        f=faces[-1]
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # crop the face largest face
        if y-offset>0 and x-offset>0:
            cropped_face = img[y-offset:y+h+offset, x-offset:x+w+offset]
        else:
            cropped_face = img[y:y+h+offset, x:x+w+offset]
        
        cropped_face = cv2.resize(cropped_face, (100, 100))
        
        skip+=1
        if skip%10==0:
            face_data.append(cropped_face)
            print("Save so far " , len(face_data))

        
    # cv2.imshow("Cropped Face", cropped_face)
    cv2.imshow("Image Window", img)
    key = cv2.waitKey(1)  # 1ms pause for every frame to show next image
    if key == ord('q'):
        break

# Write the facedata on disk
face_data=np.asarray(face_data)
print(face_data.shape)
m=face_data.shape[0]

# 13, 100, 100, 3 -> 13, 30000
face_data=face_data.reshape((m,-1))

print(face_data.shape)

# save this data into file system
file=dataset_path+name+'.npy'
np.save(file,face_data)
print("Data successfully saved at ",file)

# release camera, close window
cam.release()
cam.destroyAllWindows()
