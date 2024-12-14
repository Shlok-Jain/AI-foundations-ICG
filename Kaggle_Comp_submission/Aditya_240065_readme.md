# Sign Language Detection Model

This project involves a machine learning model designed to recognize static hand movements for sign language detection. The model is implemented using Python and TensorFlow.

## Features
- Preprocessing of input images from a CSV file.
- Data normalization.
- Splitting data into training and validation sets.
- Training a convolutional neural network for sign language recognition.
- Testing it on a test dataset and give predictions
---

## Prerequisites

1. **Requirements**:
   - Python 3.8 or higher
        - If not installed, install it from www.python.org
    
   - Should have all the required libraries installed. 
        - Tensorflow
        ```bash
        pip install tensorflow
        ```
        - Numpy
        ```bash
        pip install numpy
        ```
        - Pandas
        ```bash
        pip install pandas
        ```
        - scikit-learn
        ```bash
        pip install scikit-learn
        ```
    
    - Should have the required file *train.csv*, which acts as the training data for our project. It can be downloaded from the competition page of ICG on Kaggle.

    - Should also have the file *test.csv*, which is the testing dataset, whose predictions is being submitted. It can also be downloaded along with *train.csv* form the same page.
        

2. **Running the code**:

   - After all these requirements have been fulfilled in the environment you are working, run the cells of the python file in a sequential manner. 

   - Upon following all the given steps, you should be able to see a new csv file produced by pandas, which has the predictions of the model for each id given in the test dataset.

## Contributors
**Aditya Raj<br>
240065**
