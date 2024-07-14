import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical

# Load the DataFrame
# Adjust the path to your CSV file
df = pd.read_csv('data/tmp/encoded_train.csv')

# Separate features and target label
X = df.drop(columns=['Category']).values
y = df['Category'].values

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y, num_classes=30)  # Adjust the number of classes as needed

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential()
model.add(Dense(
    64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer='l2'))
model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
model.add(Dense(64, activation='sigmoid', kernel_regularizer='l2'))
model.add(Dense(32, activation='sigmoid', kernel_regularizer='l2'))
# 30 classes with softmax activation
model.add(Dense(30, activation='softmax'))


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Save the model
model.save('tabular_model.keras')

# Make predictions (if needed)
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
