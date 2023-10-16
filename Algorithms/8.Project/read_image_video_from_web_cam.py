import cv2

# 0: default camera
cam=cv2.VideoCapture(0)

# read image from camera object

while True:
    success, img=cam.read()

    if not success:
        print("Failed to read image from camera")
        exit(-1)

    cv2.imshow("Image Window", img)
    cv2.waitKey(1) # 1ms pause for every frame to show next image