import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from keras.utils import to_categorical

# Load the DataFrame
df = pd.read_csv('temp/train.csv')  # Adjust the path to your CSV file

# Separate features and target label
X = df.drop(columns=['Category']).values
y = df['Category'].values

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y, num_classes=39)  # Adjust the number of classes as needed

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data loaded")

# Define the model creation function
def create_model(hidden_layer_sizes=(128, 64, 32), activation='relu', alpha=0.0001, learning_rate_init=0.001):
    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0], input_dim=X_train.shape[1], activation=activation, kernel_regularizer='l2'))
    
    for size in hidden_layer_sizes[1:]:
        model.add(Dense(size, activation=activation, kernel_regularizer='l2'))
    
    model.add(Dense(39, activation='softmax'))  # Adjust according to your number of classes
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Wrap the model using KerasClassifier
model = KerasClassifier(
    model=create_model,
    epochs=50,
    batch_size=32,
    verbose=0
)

# Define the grid search parameters
param_grid = {
    'model__hidden_layer_sizes': [(128, 64, 32), (128, 64), (64, 32)],
    'model__activation': ['relu', 'tanh'],
    'model__alpha': [0.0001, 0.001],
    'model__learning_rate_init': [0.001, 0.01],
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

# Print the best parameters
print(grid_result.best_params_)
