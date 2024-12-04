import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, activations
from ucimlrepo import fetch_ucirepo
from imblearn.under_sampling import RandomUnderSampler

def main():  
    nn=84
    lr=0.005

    data = pd.read_csv(r"C:\Users\migue\OneDrive\Documentos\UCF\Machine Learning\FinalProject\Source\Data\archive\diabetes_012_health_indicators_BRFSS2015.csv")

    #best perfomance

    Features = data.drop(columns=['Diabetes_012'])
    Targets = data['Diabetes_012'] 

    #scaling features
    scalar = StandardScaler()
    x_scaled = scalar.fit_transform(Features)

    #encoding the target variable
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(Targets)
    y_cat = to_categorical(y_encoded)

    # Reshape input to be 3D [samples, timesteps, features] for LSTM
    # For simplicity, assume each record is a single timestep
    X_reshaped = np.reshape(x_scaled, (x_scaled.shape[0], 1, x_scaled.shape[1]))


    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_cat, test_size=0.2, random_state=42)


    #first model of two layers 
    model = Sequential()
    model.add(LSTM(84, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation= activations.sigmoid))
    model.compile(optimizer = optimizers.SGD(learning_rate= 0.005) , loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis = 1)
    y_true = np.argmax(y_test, axis =1)

    precision  = precision_score (y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true , y_pred, average='weighted')

    print(f'Number of Neurons: {nn:.2f} & learning rate: {lr:.4f}' )
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print("==========================================================================================")

    result = confusion_matrix(y_true , y_pred, normalize='pred')

    disp = ConfusionMatrixDisplay(confusion_matrix=result)

    disp.plot(cmap='Blues')

    plt.title('Confusion Matrix best model')

    plt.show()