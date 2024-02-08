# SAFESITE- Real-time Construction PPE Kits Analysis using YoloV8

**Team Name:** Code Catalysts <br>
**Team Members:** Amal Raj Singh, Manasvi Logani <br><br>
<div style="text-align:center;">
    <img src="https://github.com/singhamal001/SafeSite/blob/main/safesite%20logo.jpg" alt="SafeSite Project Logo" width = '400px'/>
</div> <br> <br>

## Introduction

### Importance of the Project
In the construction sector, an alarming frequency of accidents is recorded annually, many of which could be mitigated through the adoption of appropriate Personal Protective Equipment (PPE). PPE encompasses various gear,like helmets, goggles, and safety vests, designed to be worn by individuals to shield against potential health or safety hazards encountered in the workplace.

### Aim of the Project
The aim of this project is to enhance construction site safety by employing computer vision to analyze and monitor the proper usage of Personal Protective Equipment (PPE) kits in real-time. 

### About the Data
The project utilized live video feeds and images from construction sites from around Plaksha University for continuous monitoring of workers, thus utilizing the images to create our own dataset. The dataset comprised of `500 images` in jpg with the labels in YoloV8 format (txt). This dataset is split into two parts, `train: 450` and `test: 50`. <br> <br>
There are 4 classes detected by the object detection model:<br><br>
**'Hard hat', 'No Hard hat', 'Jacket', 'No Jacket'** <br>

### SetUp
The model was tarined on Google Colab 
