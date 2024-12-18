
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

label_dict = {
    0:'A' , 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',10:'K',11:'L',12:'M',
    13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y'       
}


def givepred(image_path,model):
    img_size = (28, 28)  

    img = load_img(image_path, target_size=img_size, color_mode = "grayscale")  
    img_array = img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, height, width, channel)

    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)  # Get the index with the highest probability
    predicted_label = label_dict.get(predicted_index, "Unknown")  # Map index to label
    return predicted_label