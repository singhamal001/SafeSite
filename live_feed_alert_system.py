# Import Statements
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import requests


# Discord webhook URL for sending alerts
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
    A class for performing object detection using a YOLO model and sending alerts to Discord.
    """
    def __init__(self, capture_index):
        """Initializes the object detection system.

        Parameters:
        - capture_index (int): The index of the camera to use for video capture.
        - alert_sent (bool): True if Alert is already sent in the past 10 seconds, to avoid alert Spamming."""

        # init parameters.
        self.capture_index = capture_index
        self.alert_sent = False

        # Loading the best fine-tuned model that was exported as .pt file after {}EPOCHS.
        self.model = YOLO("best_50epochs.pt")

        # init variables for FPS calculation
        self.annotator = None
        self.start_time = 0
        self.end_time = 0
        self.last_alert_time = 0

        # Using GPU if detected otherwise switching to CPU (For better FPS).
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self, im0):
        """Runs the YOLO model prediction on the given image.

        Parameters:
        - im0 (numpy.ndarray): The image on which to perform detection.

        Returns:
        - results: The detection results from the YOLO model.
        """
        results = self.model(im0, show=True, conf = 0.5, save=True)
        return results

    def display_fps(self, im0):
        """Calculates and displays the FPS on the given image frame.

        Parameters:
        - im0 (numpy.ndarray): The image frame to annotate with FPS information.

        Returns:
        None
        """
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        """Plots bounding boxes and labels for detected objects on the image frame.

        Parameters:
        - results: The detection results from the YOLO model.
        - im0 (numpy.ndarray): The image frame to annotate with detections.

        Returns:
        - im0 (numpy.ndarray): The annotated image frame.
        - class_ids (list): List of detected class IDs.
        """
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
        """Starts the object detection and alerting process.

        Returns:
        None
        """
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        # Setting The Frame size for the Webcam Live Feed.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # init Frame Count for calculating fps.
        frame_count = 0

        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret

            # detecting for availability or absence of PPE Kit in the Camera Feed.
            results = self.predict(im0)

            # Extracting Results and converting labels into int.
            ob_det = results[0].boxes.cls
            ob_det = ob_det.tolist()
            ob_det = [int(item) for item in ob_det]
            im0, class_ids = self.plot_bboxes(results, im0)

            # Looking for trigger for the alert.
            trigger = 2 in ob_det or 3 in ob_det

            # Check if more than 10 seconds have passed since the last alert
            if time() - self.last_alert_time > 10:
                self.alert_sent = False

            # If alert triggered, then sending the alert.
            if trigger and not self.alert_sent:
                message = "**⚠️ Safety Compromise Detected!**"
                send_discord_message(webhook_url, message)
                self.alert_sent = True
                self.last_alert_time = time()

            self.display_fps(im0)
            cv2.imshow('Live Cam Detection for Safety Kit', im0)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()
