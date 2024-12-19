# FSx0: FingerSpelling classification

FSx0 can process samples of video captured from the webcam in real-time and output the respective english letter spelled in ASL format.

You can find the finger spelling dictionary inside letters.png

### How it works

```mermaid
graph LR
    subgraph Process1 [Process 1: Video Capture]
        A1[Start Video Capture Process] --> A2[Open Video Stream]
        A2 --> A3[Capture Frame]
        A3 --> A4[Save Top-Left Portion of Frame]
        A4 --> A5[Check for 'q' Key Press]
        A5 -- 'q' Pressed --> A6[Stop Process 1]
        A5 -- 'q' Not Pressed --> A3
    end

    subgraph Process2 [Process 2: Image Processing]
        B1[Start Image Processing Process] --> B2[Wait for New Image]
        B2 --> B3[Load Saved Image]
        B3 --> B4[Preprocess Image: Resize & Grayscale]
        B4 --> B6[Predict Output]
        B6 --> B7[Display or Save Prediction]
        B7 --> B2
        C1[Load CNN Model with Pretrained Weights] --> B6
    end

    A4 --> B2[Image Saved for Processing]
    A6 --> B8[End]
```

### Setup

First install necessary libraries by running
```
pip install -r requirements.txt
```

### To run

``` 
cd FSx0
 ```

 To run FSx0, simply run the main.py!
``` 
python main.py
 ```

 To terminate the program, press ```q```.

 #### Disclaimer

 The CNN model may not function as intended due to variation in user's video feed from trained dataset