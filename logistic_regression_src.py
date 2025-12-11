import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class ClassifierBase():

    def __init__(self,):
        self.K = None

    def fit(self, X, y, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

    def predict(self, X_new):
        raise NotImplementedError("Subclasses should implement this!")
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)
    
    def cross_validation(self, X, y, K_cv=10, runs=1, **kwargs):
        """
        Performs K-fold cross-validation. Returns mean accuracy and standard error.
        """
        accuracies = []

        for run in range(runs):
            cv_folds = KFold(n_splits=K_cv, shuffle=True)
            for indices_train, indices_test in cv_folds.split(X):
                X_train, X_test = X[indices_train], X[indices_test]
                y_train, y_test = y[indices_train], y[indices_test]

                self.fit(X_train, y_train, **kwargs)
                accuracies.append(self.score(X_test, y_test))

        mean_accuracy = np.mean(accuracies)
        std_error = np.std(accuracies, ddof=1) * np.sqrt(1/ K_cv / runs)
        return mean_accuracy.item(), std_error.item()
        
    def confusion_matrix(self, X, y, K_cv=10, runs=10, class_labels=None, **kwargs):
        """
        Estimates the confusion matrix using K-fold cross-validation.
        Returns a DataFrame with predicted vs actual classes.
        """
        if self.K is None:
            self.K = np.unique(y).shape[0]
        df = pd.DataFrame(np.zeros((self.K, self.K), dtype=int), index=class_labels, columns=class_labels)
        df.columns.name = 'Actual class:'
        df.index.name = 'Predicted class:'

        for run in range(runs):
            cv_folds = KFold(n_splits=K_cv, shuffle=True)
            for indices_train, indices_test in cv_folds.split(X):
                X_train, X_test = X[indices_train], X[indices_test]
                y_train, y_test = y[indices_train], y[indices_test]

                self.fit(X_train, y_train, **kwargs)
                y_pred = self.predict(X_test)

                for pred, actual in zip(y_pred.flatten(), y_test.flatten()):
                    df.iloc[pred, actual] += 1

        # Normalize confusion matrix to fractions (row-wise: predicted class)
        df = df.div(df.sum(axis=0), axis=1).fillna(0).round(2)

        return df

    def plot(self, X, y, feature_indices=(0, 1), directions=None, title=None, labels=None):

        colors = ['red', 'blue', 'lightgreen']
        points = 200
        if self.K is None:
            self.K = np.unique(y).shape[0]
        
        # Set directions
        if directions is None:
            directions = np.zeros((X.shape[1], 2))
            directions[feature_indices[0], 0] = 1.
            directions[feature_indices[1], 1] = 1.
        else:
            directions = np.array(directions)
            directions = directions[:, :2] # Take only first two columns
            assert directions.shape == (X.shape[1], 2), "Directions must be of shape (n_features, 2)"
            directions = directions / np.linalg.norm(directions, axis=0) # Normalize directions

        X = X @ directions
        
        # Select features for plotting
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        # # Create meshgrid for prediction regions
        x_range = np.linspace(x_min, x_max, points)
        y_range = np.linspace(y_min, y_max, points)
        xx, yy = np.meshgrid(x_range, y_range)

        X_grid = np.outer(xx, directions[:,0]) + np.outer(yy, directions[:,1])
        y_grid_pred = self.predict(X_grid).reshape(points,points)

        plt.figure(figsize=(5, 5))
        plt.contourf(x_range, y_range, y_grid_pred, 2, alpha=0.2, colors=colors)

        # Plot training points
        for k in range(self.K):
            plt.scatter(X[:, 0][y.flatten() == k], X[:, 1][y.flatten() == k],
                        c=colors[k], label=labels[k] if labels is not None else f"Class {k}")

        plt.xlabel(f"Direction {directions[:,0].round(2)}")
        plt.ylabel(f"Direction {directions[:,1].round(2)}")
        if title:
            plt.title(title)
        plt.legend()
        plt.show()

def plot_digits_sample(X, y):
    N = len(y)
    _, axes = plt.subplots(nrows=3, ncols=4, figsize=(8, 5))
    for ax in axes.flatten():
        idx = np.random.randint(N) 
        image = X[idx]
        label = y[idx, 0]

        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Target: %i" % label)

    plt.show()