from ultralytics import YOLO
import cv2

# #Loading Model
model = YOLO("best_50epochs.pt") #building new model from scratch

result = model.predict("datasets/test/images/img (59).jpg")

