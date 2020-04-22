#!/usr/bin/env python3

import os 

from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract

import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def decode_predictions(scores, geometry):
    	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < 0.95:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def text_detect(image_path, net):
    im_list = [file for file in os.listdir(image_path) if file.endswith('.JPG')]
    im_num = len(im_list)
    results = {}
    text_detect_fails = []
    text_recognition_fails = []
    for i, im in enumerate(im_list):
        print(f"currently on {i+1} / {im_num} iteration - {im}")
        path = os.path.join(f"{image_path}", im)
        img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
        orig = img
        orig_h, orig_w = orig.shape[:2]
        (newW, newH) = (3008, 4032)
        rW = orig_w / float(newW)
        rH = orig_h / float(newH)

        # resize the image and grab the new image dimensions
        img = cv2.resize(img, (newW, newH))
        (H, W) = img.shape[:2]

        layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
        blob = cv2.dnn.blobFromImage(img, 1.0, (img.shape[1], img.shape[0]))
        net.setInput(blob)
        scores, geometry = net.forward(layerNames)
        print('Image passed through network.')
        rects, confidences = decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        if len(boxes) == 0:   
            text_detect_fails.append(im)
            continue
        likely_id = ""
        for startX, startY, endX, endY in boxes:
            startX = int(startX*rW)
            startY = int(startY*rH)
            endX = int(endX*rW)
            endY = int(endY*rH)
            dX = int((endX - startX) * 0.2)
            dY = int((endY - startY) * 0.2)

            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(orig_w, endX + (dX * 2))
            endY = min(orig_h, endY + (dY * 2))        

            roi = orig[startY:endY, startX:endX]
            text = pytesseract.image_to_string(roi, config="-l eng --oem 3 --psm 7 digits")
            if len(text) == 2:
                likely_id = text
                cv2.imwrite(f'{im}_identified.PNG', roi)       
                print(f'IDENTIFIED! - {text}')
        if len(likely_id) == 0:
            text_recognition_fails.append(im)
            continue
        results[im[4:8]] = text
    return results, text_detect_fails, text_recognition_fails

if __name__ == '__main__':
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')
    print('Read in text detection network')
    res, det_fail, rec_fail = text_detect('./IndividualImages1/', net=net)
    print(res)
    print(det_fail, rec_fail)
    with open('results_1.txt', 'w') as f:
        for k, v in res.items():
            f.write(str(k)+" "+str(v) + "\n")

    