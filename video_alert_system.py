# Import Statements
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import requests


# Define the Discord webhook URL for sending alerts
webhook_url = "https://discord.com/api/webhooks/1204840755126992976/d3yDKKt-Q-2dU0rbk34XDNQXE8c22uSDuSpoJwEXShKWZlwaIU9p6FRwmgH47byJuj-0"
def send_discord_message(webhook_url, message):
    """Sends a message to a Discord channel using the specified webhook URL.

    Parameters:
    - webhook_url (str): The webhook URL to post the message to.
    - message (str): The message content to send.

    Returns:
    None"""

    data = {"content": message}
    response = requests.post(webhook_url, json=data)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"Failed to send message: {err}")

class ObjectDetection:
    """
    A class for performing object detection on video files using a YOLO model and sending alerts to Discord.
    """
    def __init__(self, video_path):
        """Initializes the object detection system with a video file.

        Parameters:
        - video_path (str): The path to the video file to use for detection.
        """
        self.video_path = video_path
        self.alert_sent = False

        # Load the YOLO model.
        self.model = YOLO("best.pt")

        # Initialize variables for annotation.
        self.annotator = None

        # Using GPU if available, otherwise CPU.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self, im0):
        """Runs the YOLO model prediction on the given frame."""
        results = self.model(im0, show=True, conf=0.85, save=True)
        return results

    def plot_bboxes(self, results, im0):
        """Plots bounding boxes and labels for detected objects."""
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return im0, class_ids

    def __call__(self):
        """Starts the object detection process on the video."""
        # Replace camera capture with video file capture.
        cap = cv2.VideoCapture(self.video_path)
        assert cap.isOpened(), "Failed to open video file"

        while True:
            ret, im0 = cap.read()
            if not ret:
                break  # Exit the loop if the video ends

            results = self.predict(im0)
            ob_det = results[0].boxes.cls
            ob_det = ob_det.tolist()
            ob_det = [int(item) for item in ob_det]
            im0, class_ids = self.plot_bboxes(results, im0)

            # Alert logic remains the same.
            trigger = 2 in ob_det or 3 in ob_det
            if trigger and not self.alert_sent:
                send_discord_message(webhook_url, "**⚠️ Safety Compromise Detected!**")
                self.alert_sent = True
                # Reset alert after a specific condition or time.

            cv2.imshow('Object Detection in Video', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage with a video file path.
video_path = 'videos/video (3).MOV'
detector = ObjectDetection(video_path)
detector()
