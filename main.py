from imblearn.over_sampling import SMOTE
import MLPModule
from cnn import CNNCode
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import transformer_model  # Import the transformer model



def evaluation(y_pred, y_true):
    print("Displaying Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()

    print("Displaying Classification Report...")
    print(classification_report(y_true, y_pred))

def mlp(X_train, y_train):
    X_train_s, y_train_s = X_train, y_train
    cols = X_train.shape[1]

    params = {'solver': 'adam',
              'activation': 'tanh',
              'alpha': 1e-4,
              'hidden_layer_sizes': (int((cols + 3)/2),),
              'learning_rate_init': 0.001,
              'max_iter': 1000,
              'random_state': 42}

    print("\n Do you want to use oversampling?")
    print("1. Yes (Recommended).")
    print("2. No (Not Recommended).")

    while True:
        selection = input("Enter your choice of oversampling: ")
        if selection == "1":
            print("You've chosen to use oversampling!")
            dict = {'0.0': 200000, '1.0': 600000, '2.0': 400000}
            smote = SMOTE(sampling_strategy=dict, random_state=42)
            X_train_s, y_train_s = smote.fit_resample(X_train, y_train)
            break
        elif selection == "2":
            print("You've chosen to not use oversampling.")
            break
        else:
            print("Invalid Input, try again!")

    print("\nHow do you want your classifier defined?")
    print("1. Using Predetermined Parameters (Recommended)?")
    print("2. Doing a Grid Search (WARNING: May Take Well Over A Day!)?")
    print("3. Default Parameters (WARNING: Inaccurate Classifier)?")
    print("0. Cancel MLP -- Exit")

    while True:
        selection = input("Enter your choice of hyperparameter tuning: ")
        if selection == "1":
            print("You've chosen to use predetermined parameters...")
            mlp_classifier = MLPModule.MLP(X_train_s, y_train_s, params=params)
            return mlp_classifier
        elif selection == "2":
            print("You've chosen to do a Grid Search...")
            mlp_classifier = MLPModule.MLP(X_train_s, y_train_s, search=True)
            return mlp_classifier
        elif selection == "3":
            print("You've chosen to use Default Parameters...")
            mlp_classifier = MLPModule.MLP(X_train_s, y_train_s)
            return mlp_classifier
        elif selection == "0":
            print("Exiting MLP...")
            return None
        else:
            print("Invalid Input, try again!")

def main():
    df = pd.read_csv('./diabetes_012_health_indicators_BRFSS2015.csv')

    X_train, X_val_test, y_train, y_val_test = train_test_split(df.iloc[:, 1:], df['Diabetes_012'].astype(str), test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

    print("\nAvailable Classifiers: ")
    print("1. Fine-Tuned Large Language Model (LLM)")
    print("2. Convolutional Neural Network (CNN)")
    print("3. Multi-Layer Perceptron (MLP)")
    print("4. Long Short-Term Memory (LSTM)")
    print("5. Transformer")
    print("0. Exit")

    while True:
        choice = input("Enter your choice of classifier: ")

        if choice == "1":
            print("IMPLEMENT LATER")
            break
        elif choice == "2":
            print("\nAvailable CNN models: ")
            print("1. Complex model")
            print("2. Basic Model")
            print("0. Exit")

            while True:
                choice = input("Enter your choice of model: ")
                if choice == "1":
                    CNNCode(1,X_train,y_train,X_test,y_test,X_val,y_val)
                if choice == "2":
                    CNNCode(2,X_train,y_train,X_test,y_test,X_val,y_val)
                elif choice == "0":
                    print("Exiting...")
                    return
                return
            break
        elif choice == "3":
            mlp_classifier = mlp(X_train, y_train)
            if mlp_classifier == None:
                print("Returning to Main Menu...")
            
            y_pred = mlp_classifier.predict_classifier(X_train)
            print("Evaluation Confusion Matrix and Classification Report for Training Set...")
            evaluation(y_pred, y_train)

            y_pred = mlp_classifier.predict_classifier(X_val)
            print("Evaluation Confusion Matrix and Classification Report for Validation Set...")
            evaluation(y_pred, y_val)

            y_pred = mlp_classifier.predict_classifier(X_test)
            print("Evaluation Confusion Matrix and Classification Report for Testing Set...")
            evaluation(y_pred, y_test)

            return
        elif choice == "4":
            print("IMPLEMENT LATER")
            break
        elif choice == "5":
            transformer_model.main(X_train,y_train,X_test,y_test,X_val,y_val)            
            break
        elif choice == "0":
            print("Exiting...")
            return
        else:
            print("Invalid Input, try again!")

if __name__ == "__main__":
    main()
