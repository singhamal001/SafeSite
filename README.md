# SAFESITE- Real-time Construction PPE Kits Analysis using YoloV8

**Team Name:** Code Catalysts 

**Team Members:** Amal Raj Singh, Manasvi Logani 





<div style="text-align:center;">
    <img src="https://github.com/singhamal001/SafeSite/blob/main/safesite%20logo.jpg" alt="SafeSite Project Logo" width="400px" />
</div>




## Introduction
### Importance of the Project
In the construction sector, an alarming frequency of accidents is recorded annually, many of which could be mitigated through the adoption of appropriate Personal Protective Equipment (PPE). PPE encompasses various gear,like helmets, goggles, and safety vests, designed to be worn by individuals to shield against potential health or safety hazards encountered in the workplace.

### Aim of the Project
The aim of this project is to enhance construction site safety by employing computer vision to analyze and monitor the proper usage of Personal Protective Equipment (PPE) kits in real-time. 

### About the Data
The project utilized live video feeds and images from construction sites from around Plaksha University for continuous monitoring of workers, thus utilizing the images to create our own dataset. The dataset comprised of `500 images` in jpg with the labels in YoloV8 format (txt). This dataset is split into two parts, `train: 450` and `test: 50`. Due to the confidentiality of the data, the images cannot be shared here.

There are 4 classes detected by the object detection model:

**'Helmet', 'No Helmet', 'Safety Vest', 'No Safety Vest'** 

<div style="text-align:center;">
    <img src="https://github.com/singhamal001/SafeSite/blob/main/5100571.jpg" alt="Jacket and Helmet Image" width="400px" />
</div>

>Image by <a href="https://www.freepik.com/free-vector/engineering-construction_12893261.htm#query=construction%20ppe&position=20&from_view=keyword&track=ais&uuid=b6cf4565-76c3-4e80-91ba-8ef52bdc13d0">Freepik</a>

### SetUp
The model was trained on Google Colab, utilizing **5.8 G VRAM per epoch** for 50 epochs. Other code laying out the *'Live feed alert system'* and *'Video alert system'* was directly run on the local system. 

### File Heirarchy
1. `Data_CV` is the main folder consisting of the yaml file *config.yaml* required for training. It further contains 2 folders `train` and `test`. Each of these two folders have two subfolders `images` ( .jpg files) and `labels` (.txt files with annotations).

2. Under the `Data_CV` folder, the `models` folder is present comprising of `yolo8n.pt`which is the pretrained model and the `best_50epochs.pt` and `best_85epochs.pt` for the custom trained **Yolo8vn** model on our custom dataset.

3. The `runs` folder contains all the live tracking results of the video tracking run on the local computer. 

4. Finally, `best_val_results` conatins the confusion matrices, F1 curve, Yolo metrics graphs and PR curve.

### Code-based Heirarchy

```
├───Data_CV
├───├──data.yaml
│   ├───test
│   │   ├───images
│   │   └───labels
│   ├───train
│       ├───images
│       └───labels
│ 
│       
├───models
├───runs
│   └───video results
├───best_val_results
    └───graphs
```
### Packages Used
- `torch`: PyTorch is an open-source machine learning framework that provides tensors and dynamic computational graphs for building deep learning models.
- `numpy`: NumPy is a fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
- `opencv-python`: OpenCV is a library of programming functions mainly aimed at real-time computer vision, offering support for image and video analysis, object detection, and various image processing tasks.
- `time`: The time module in Python provides various time-related functions, including time measurement, conversion, and manipulation, facilitating tasks such as performance analysis and scheduling.
- `ultralytics`: Ultralytics is a Python library offering implementations of state-of-the-art deep learning models for computer vision tasks, with a focus on object detection and image classification.
- `requests`: Requests is a simple yet powerful HTTP library for Python, enabling users to make HTTP requests easily, handle responses, and interact with web services and APIs.
- `pillow`: Pillow is the friendly PIL (Python Imaging Library) fork, providing support for opening, manipulating, and saving many different image file formats, making it a versatile tool for image processing tasks.


### Methodology

1. **Dataset Creation**:
   We generated a novel dataset from construction sites surrounding the campus, comprising images categorized into four classes: 'No safety vest', 'No helmet', 'Helmet', and 'Safety Vest'.

2. **Image Annotation**:
   Utilizing the annotation platform provided by [MakeSense AI](https://www.makesense.ai/), we meticulously annotated all images. Annotations were exported to .txt files, conforming to the labeling format required for training YOLOv8.

3. **Model Selection**:
   Considering computational efficiency, we opted for the nano model variant of YOLOv8, denoted as YOLOv8n, which suits our system specifications.

4. **Model Training**:
   Training the YOLOv8 model was conducted on Google Colab. We experimented with training durations of 50 and 85 epochs, exporting the optimal settings as 'best_50epochs.pt' and 'best_85epochs.pt' respectively.

```Python
!yolo task=detect mode=train model=yolov8n.pt data= data.yaml epochs=10 imgsz=640 plots=True

```

5. **Integration with Live Feed**:
   Post-training, the model settings were integrated into our local environment, enabling real-time processing of the live webcam feed from the default camera.

6. **Alert System Development**:
   We initiated the development of an alert system aimed at detecting safety compromises. Initially, the plan involved email alerts upon detection. However, due to challenges with SMTP server configurations and authentication errors with Gmail, we transitioned to Discord Webhooks.

7. **Discord Webhooks Integration**:
   Discord Webhooks emerged as an alternative to email alerts, offering seamless API-driven communication. We established a trigger system to identify labels detected in each frame from the webcam feed.

8. **Safety Compromise Detection**:
   By extracting class IDs from the YOLO model predictions, we identified safety-related objects in the webcam feed, mapped to integer values corresponding to the four classes. Any detection of class IDs representing 'No helmet' or 'No safety vest' prompted an immediate alert to the designated Discord server's Alert Channel.

9. **Preventative Measures**:
   To mitigate spam alerts, we implemented a 10-second timeout mechanism between consecutive alerts. This ensures that alerts are dispatched judiciously, preventing unnecessary disruption.

## Results

### Confusion Matrix

<div style="text-align:center;">
    <img src="https://github.com/singhamal001/SafeSite/blob/main/val_results/confusion_matrix_normalized.png" alt="Confusion Matrix" width="450px" />
</div>

The confusion matrix demonstrates promising results for the classification model, particularly in its ability to recognize **'Helmet'** and **'Safety Vest'** with high accuracy, showcasing true positive rates of **90%** and **93%**, respectively. These results underscore the model's robustness in identifying essential safety equipment, a critical component in ensuring construction site safety.

Notably, the model also exhibits a reasonable degree of accuracy in identifying **'No Helmet'** and **'No Safety Vest'** classes, with true positive rates of **78%** and **79%**.

But there are areas where the model's performance can be further refined, such as reducing the misclassification between similar classes. The most significant reason for this is imbalanced data as there were less instances for 'No Helmet' and 'No Safety Jacket'.

### Training/Validation Metrics Graph

<div style="text-align:center;">
    <img src="https://github.com/singhamal001/SafeSite/blob/main/val_results/results.png" alt="Training/Validation Metrics" width="450px" />
</div>

1. **Loss Metrics (train/box_loss, train/cls_loss, train/dfI_loss, val/box_loss, val/cls_loss, val/dfI_loss)**:

- All three training loss metrics (box, class, and dfI) show a clear downward trend, indicating that the model is learning and improving its predictions over epochs.
- The validation loss metrics are more volatile, especially box loss, but they do trend downward, which suggests the model is generalizing well to unseen data. 

2. **Precision and Recall (metrics/precision(B), metrics/recall(B))**

 - Precision starts high and remains relatively stable throughout training, which indicates that when the model predicts an object class, it is often correct.
- Recall shows significant improvement over time, which suggests the model is getting better at identifying all relevant objects in the dataset.

3. **Mean Average Precision (metrics/mAP50(B), metrics/mAP50-95(B))**

 - The mAP at Intersection over Union (IoU) threshold of 0.5 (mAP50) is high and shows an upward trend, indicating good model performance at this particular IoU threshold.
- The mAP across IoU thresholds from 0.5 to 0.95 (mAP50-95) also increases over time, though it starts much lower than mAP50. This metric is a stricter measure of model performance and its improvement suggests that the model is becoming better at precisely localizing objects across different IoU thresholds.

### F1-Confidence Curve

<div style="text-align:center;">
    <img src="https://github.com/singhamal001/SafeSite/blob/main/val_results/F1_curve.png" alt="F1-Confidence Curve" width="450px" />
</div>

- The 'Helmet' class maintains the highest F1 score across a wide range of confidence thresholds, which indicates a strong balance between precision and recall — the model is consistently accurate and reliable in identifying helmets.
- The 'Safety Vest' class also shows a high F1 score, though it slightly underperforms compared to the 'Helmet' class. This suggests good model performance but with a little more room for improvement, either in precision or recall, or both.
- The 'No Safety Vest' and 'No Helmet' classes have lower F1 scores across confidence thresholds. The F1 score for 'No Safety Vest' is notably lower than 'Safety Vest', suggesting that the model is less adept at recognizing the absence of a safety vest. This may be due to the mentioned data imbalance or more complex visual patterns when a safety vest is not present.
- The 'No Helmet' class shows a similar pattern to 'No Safety Vest', with a lower F1 score compared to the 'Helmet' class, indicating more difficulty in accurately identifying instances where a helmet is not present.
- The F1 score for all classes combined reaches its peak at a confidence threshold of approximately 0.2, with an F1 score of 0.83. This peak represents the optimal balance between precision and recall across all classes at this particular confidence threshold.
- As the confidence threshold increases, the F1 scores for all classes generally decrease, which is typical as the model becomes more conservative and predicts fewer objects, potentially missing some true positives.

### Recall-Confidence Curve

<div style="text-align:center;">
    <img src="https://github.com/singhamal001/SafeSite/blob/main/val_results/R_curve.png" alt="Recall-Confidence" width="450px" />
</div>

- Across all confidence thresholds, the 'Helmet' class maintains the highest recall, indicating that the model is consistently good at identifying all relevant helmet instances in the dataset.
- The 'Safety Vest' class also demonstrates high recall, albeit slightly lower than the 'Helmet' class, suggesting that the model is quite effective at detecting safety vests, with minimal instances of missed detections.
- The recall for 'No Safety Vest' and 'No Helmet' starts comparably lower and decreases more sharply with the increase in confidence threshold. This could indicate that the model has more false negatives for these classes, possibly due to a variety of factors such as class imbalance or the inherent difficulty in detecting the absence of an item.
- The recall for all classes combined is at its peak when the confidence threshold is at its lowest, which is expected as the model is less selective and therefore captures more true positives.
- However, as the confidence threshold increases, recall decreases across all classes. This is typical, as a higher threshold means the model is more conservative in predicting objects, thus potentially missing some true positives.

### Precision-Confidence Curve

<div style="text-align:center;">
    <img src="https://github.com/singhamal001/SafeSite/blob/main/val_results/P_curve.png" alt="Precision-Confidence Curve" width="450px" />
</div>

- The 'Helmet' class has consistently high precision across most confidence thresholds, suggesting that when the model predicts an instance as 'Helmet', it is very likely to be correct.
- The 'Safety Vest' class also shows high precision, though it experiences a slight dip as the confidence threshold increases. This dip may indicate a few false positives at higher thresholds.
- The 'No Safety Vest' class has lower precision at lower confidence thresholds, which improves as the threshold increases. This suggests the model becomes more accurate in its predictions for this class as it becomes more confident, but at the cost of potentially missing some correct detections (lower recall).
- Precision for the 'No Helmet' class follows a similar trend to 'No Safety Vest', but with a more pronounced drop at higher thresholds, which might indicate a higher rate of false positives for uncertain detections.
- Across all classes, the precision starts very high at low confidence thresholds, indicating a large number of true positives among the detections. However, there is a slight decline in precision as the confidence threshold increases, which is typical because the model makes fewer, but more confident, predictions.
- The graph indicates an optimal point where precision for all classes peaks at a confidence threshold near 0.981.

### Precision-Recall Curve

<div style="text-align:center;">
    <img src="https://github.com/singhamal001/SafeSite/blob/main/val_results/PR_curve.png" alt="Precision Recall Curve" width="450px" />
</div>

- The 'Helmet' class shows the highest precision across all levels of recall, with an mAP of 0.965. This excellent performance indicates that the model is very accurate in identifying helmets and rarely misses any instances.
- The 'Safety Vest' class also has a high precision-recall balance, with an mAP of 0.926. This suggests that the model is quite reliable in detecting safety vests, with both high precision and recall.
- The 'No Safety Vest' class shows moderate precision and recall with an mAP of 0.840. While the performance is good, it is not as high as the 'Safety Vest' class, which could be due to factors like data imbalance or the inherent difficulty of detecting the absence of an item.
- The 'No Helmet' class has the lowest precision and recall curve, with an mAP of 0.758. This indicates that identifying the absence of a helmet is the most challenging task for the model, possibly requiring additional training data or refinement of features.
- The curve for all classes combined has an mAP of 0.872 at IoU 0.5, reflecting a strong overall performance of the model.


## Conclusion

In conclusion, the safety monitoring model developed for construction sites demonstrates impressive performance in detecting the presence of safety equipment, such as helmets and safety vests, with high precision and recall. It also provides a satisfactory ability to identify the absence of such equipment, although with a slightly lower accuracy, suggesting further refinement could be beneficial.

The dataset, annotations, and careful selection of the YOLOv8n model have resulted in a robust system capable of real-time processing and alerting via Discord Webhooks, with preventative measures to reduce alert fatigue. Overall, the model achieves a good mean Average Precision across all classes, indicating its reliability and effectiveness as a tool for enhancing safety compliance on construction sites.

Future work will focus on addressing the challenges of class imbalance and improving detection accuracy for the less represented classes to ensure the model's continued improvement and adaptability to diverse construction environments.



## References

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)
