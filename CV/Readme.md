# Facial recognition using Resnet, SVM, RF
## Requirements 
- Python 3.7.2 
## Usage
- Begin by starting a virtual environment and installing from the requirements.txt file:
```bash
pip3 install -r requirements.txt
```
Then clone the directory and download the models folder into the same directory (must be present for `recognise_face.py` to work).
Then you can run the following commands based on the image path `im_path`:
```bash
python3 recognise_face.py im_path classifier_type --features extractor_type --creative_mode 
```
Possible classifier_type values to use in this function are:
- cnn, svm, rf 

Possible extractor_type values are:
- sift, surf

The `--features` call is optional, and required only when `classifier_type = svm ` or `classifier_type = rf` 
Similarly, the `--creative_mode` call is optional - calling it cartoonifies the faces in the test image.

## Included modules and scripts
This section is a brief overview of what each module included entails:
Model training scripts and notebooks:
- `SVM_RF_train.ipynb`
- `retrain_resnet.ipynb`
- `retrain_resnet.py`

Face detection script: 
- `face_detector.py`
- All files in retinaface directory

Text detection & OCR script:
- `text_detector.py`

Creative mode:
- `creative.py`

Face recognition scripts:
- `feature_extractors.py`
- `recognise_face.py`

## Outputs
In the command line, a list of `ID - x_loc, y_loc` are presented. For `classifier_type = cnn` these will be presented in decreasing probability order. These predictions are also written to `results.txt`. The classified picture with bounding boxes and ID predictions overlaid is also written to `results{time}.jpg`. This file will also have any creative mode changes applied to it as called.
