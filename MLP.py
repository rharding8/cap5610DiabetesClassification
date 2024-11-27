from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class MLP:
    def __init__(self, X_train, y_train, params=None):
        if params != None:
            self.classifier = MLPClassifier(**params)
            self.fit_classifier(X_train, y_train)
        
        self.find_set_best_params(X_train, y_train)

    def find_set_best_params(self, X_train, y_train):
        cols = X_train.shape[1]
        params = {'solver': ['sgd', 'adam'],
                    'activation': ['relu', 'logistic', 'tanh'],
                    'alpha': [1e-5, 5e-5, 1e-4, 5e-4, 0.001],
                    'hidden_layer_sizes': [((cols + 1)/2,), ((cols + 3)/2,), (100,)],
                    'learning_rate_init': [1e-4, 5e-4, 0.001, 0.005, 0.01],
                    'max_iter': [500, 1000, 1500]}

        clf = MLPClassifier(random_state=42)
        grid = GridSearchCV(estimator=clf, param_grid=params, scoring='f1_weighted')
        grid.fit(X_train, y_train)
        self.classifier = grid.best_estimator_

    def fit_classifier(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict_classifier(self, X_test):
        return self.classifier.predict(X_test)