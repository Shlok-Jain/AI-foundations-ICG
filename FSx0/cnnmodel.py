import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,Flatten, Dropout, Dense, Rescaling

IMG_SHAPE = (28,28,1)

aug_layers = tf.keras.models.Sequential([
    Input(IMG_SHAPE),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1,0.1)
    
])

model = tf.keras.models.Sequential([
    Input(shape=IMG_SHAPE),
    Rescaling(1./255),
    
    Conv2D(16,(3,3),padding = 'same',activation= 'relu'),
    
    Conv2D(64,(3,3),padding = 'same',activation= 'relu'),
    
    Conv2D(128,(3,3),padding = 'same',activation= 'relu'),
    MaxPooling2D(),
    
    Conv2D(256,(3,3), activation='relu'),
    MaxPooling2D(),
    
    Flatten(),
    
    Dense(128,activation= 'relu'),
    Dense(25, activation='softmax')
])

model_with_aug = tf.keras.models.Sequential([
    aug_layers,
    model
])

model_with_aug.compile('adam', loss=tf.keras.losses.categorical_crossentropy, metrics= ['accuracy'])

def loadCNN():
    model_with_aug.load_weights('CNN/my_checkpoint.weights.h5')
    return model_with_aug

