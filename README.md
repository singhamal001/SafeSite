# SAFESITE- Real-time Construction PPE Kits Analysis using YoloV8

**Team Name:** Code Catalysts 

**Team Members:** Amal Raj Singh, Manasvi Logani 





<div style="text-align:center;">
    <img src="https://github.com/singhamal001/SafeSite/blob/main/CV-Project_final_logo.png" alt="SafeSite Project Logo" width="500px" />
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
## Methodology

### Methodology

1. **Dataset Creation**:
   We generated a novel dataset from construction sites surrounding the campus, comprising images categorized into four classes: 'No safety vest', 'No helmet', 'Helmet', and 'Safety Vest'.

2. **Image Annotation**:
   Utilizing the annotation platform provided by [MakeSense AI](https://www.makesense.ai/), we meticulously annotated all images. Annotations were exported to .txt files, conforming to the labeling format required for training YOLOv8.

3. **Model Selection**:
   Considering computational efficiency, we opted for the nano model variant of YOLOv8, denoted as YOLOv8n, which suits our system specifications.

4. **Model Training**:
   Training the YOLOv8 model was conducted on Google Colab. We experimented with training durations of 50 and 85 epochs, exporting the optimal settings as 'best_50epochs.pt' and 'best_85epochs.pt' respectively.

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


