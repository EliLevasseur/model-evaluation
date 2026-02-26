"""Module specified for evaluating Model Performance."""

class Model:
    def __init__(self, model):
        """Initializes the Model class with a given model."""
        self.model = model
        self.results = {}


    def __repr__(self):
        return f"{self.model=}"


    def sensitivity(self, matrix):
        """Calculates the sensitivity (true positive rate) from a confusion matrix."""
        tp = matrix.iloc[1, 1]
        fn = matrix.iloc[1, 0]
        sensitivity = tp / (tp + fn)
        self.results['sensitivity'] = sensitivity


    def specificity(self, matrix):
        """Calculates the specificity (true negative rate) from a confusion matrix."""
        tn = matrix.iloc[0, 0]
        fp = matrix.iloc[0, 1]
        specificity = tn / (tn + fp)
        self.results['specificity'] = specificity
    

    def accuracy(self, matrix):
        """Calculates the accuracy from a confusion matrix."""
        tp = matrix.iloc[1, 1]
        tn = matrix.iloc[0, 0]
        fp = matrix.iloc[0, 1]
        fn = matrix.iloc[1, 0]
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        self.results['accuracy'] = accuracy
    

    def error_rate(self, matrix):
        """Calculates the error rate from a confusion matrix."""
        tp = matrix.iloc[1, 1]
        tn = matrix.iloc[0, 0]
        fp = matrix.iloc[0, 1]
        fn = matrix.iloc[1, 0]
        error_rate = (fp + fn) / (tp + tn + fp + fn)
        self.results['error_rate'] = error_rate


    def get_results(self):
        """Returns all computed metric values."""
        return self.results.copy()

