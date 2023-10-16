# Project Face Recognition

1. Collect data of various persons

   1. Asking multiple people to come in front of webcam, click 20 pictures each
   2. Store the part of image containing the fact (Harcascade to detect the face)

2. Train a classiefier to learn who is the person (Classification)
3. Predict the name of the person

   1. Read the video stream
   2. Extract the face out of it
   3. Predict the label for that face
      1. logistic regression (parametric algorithm)
      2. neural network
      3. k-nn (non parametric : look for similairity in nearest neighbors)
