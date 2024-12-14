# README.md

### Steps to Run the program.

#### 1. Upload Files
- Open the Mayank_240641_code.ipynb file in Google Colab.
- Upload `train.csv` and `test.csv` using the following code:

```python
from google.colab import files
uploaded = files.upload()
```

#### 2. Import Required Libraries
```python
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
```

#### 3. Load and Preprocess Data
- Read the `train.csv` file:

```python
train = pd.read_csv('train.csv')
X = train.iloc[:, 1:785]  # Features (pixel values)
y = train.iloc[:, 0]      # Labels
```

- Check the unique labels:

```python
print(np.unique(y))
```

- Split the data into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- Convert the data into NumPy arrays and reshape it:

```python
X_train = X_train.to_numpy().reshape(-1, 28, 28, 1) / 255  # Normalize values
X_test = X_test.to_numpy().reshape(-1, 28, 28, 1) / 255    # Normalize values
```

- One hot encoding the labels:

```python
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
```

#### 4. Build the CNN Model
- Define the CNN architecture:

```python
from tensorflow.keras import layers, models

model = models.Sequential([
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(24, activation='softmax'))
model.summary()
```

- Compile the model:

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

#### 5. Train the Model
```python
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
```

#### 6. Plotting accuracy metrics

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Training History')
plt.show()
```

#### 7. Making predictions using test data

```python
test = pd.read_csv('test.csv')
X_test_1 = test.iloc[:, 1:785].to_numpy().reshape(-1, 28, 28, 1) / 255  # Normalize
Y_pred = model.predict(X_test_1).argmax(axis=1)
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
y_pred_labels = [labels[x] for x in Y_pred]
```

#### 8. Creating submission file and downloading it

```python
submission = pd.DataFrame({'id': range(len(y_pred_labels)), 'label': y_pred_labels})
submission.to_csv('submission.csv', index=False)

from google.colab import files
files.download('submission.csv')
```
