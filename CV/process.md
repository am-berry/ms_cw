## Labelling data

1. Use YOLO retrained on faces to find the faces in the individual photos/frames of the videos
2. Use YOLO retrained on paper (possibly) to find the pieces of paper 
2.1 Use OCR to extract the number from detected pieces of paper

## Training data

1. CNN - retrain some resnet (etc) on the labelled training data (validate on some split of the data <figure this out>)
2. SURF/SIFT - extract features, train SVM/other models on the labelled training data (validate ...)

## Group images
 
1. Use YOLO retrained on faces to find the faces in the group photographs 
2. Use our CNN/SVM etc to predict on what each face is.
3. Person can only appear in once in a picture (obviously), so base on probability 

## Functions


