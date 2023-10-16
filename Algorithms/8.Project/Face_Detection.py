import cv2

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

    for f in faces:
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Image Window", img)
    key = cv2.waitKey(1)  # 1ms pause for every frame to show next image
    if key == ord('q'):
        break

# release camera, close window
cam.release()
cam.destroyAllWindows()
