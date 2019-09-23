import argparse
import dlib
import cv2
import FaceAligner
import imutils
from FaceAligner import rect_to_bb
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_detector"])
fa = FaceAligner(predictor, desiredFaceWidth=256)

image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Input", image)
rects = detector(gray, 2)

for rect in rects:
    (x, y, w, h) = rect_to_bb(rect)
    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
    faceAligned = fa.align(image, gray, rect)
    cv2.imshow("Original", faceOrig)
    cv2.imshow("Aligned", faceAligned)
    cv2.waitKey(0)