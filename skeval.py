"""Module specified for evaluating Model Performance."""

class Model:
    def __init__(self, matrix, model="Decision Tree"):
        """Initializes the Model with a confusion matrix, optional name,"""
        self.model = model
        self.matrix = matrix
        self.results = {}


    def __repr__(self):
        return f"{self.model=}"


    def sensitivity(self):
        """Calculates the sensitivity (true positive rate) from the stored confusion matrix."""
        tp = self.matrix.iloc[1, 1]
        fn = self.matrix.iloc[1, 0]
        sensitivity = tp / (tp + fn)
        self.results['sensitivity'] = sensitivity


    def specificity(self):
        """Calculates the specificity (true negative rate) from the stored confusion matrix."""
        tn = self.matrix.iloc[0, 0]
        fp = self.matrix.iloc[0, 1]
        specificity = tn / (tn + fp)
        self.results['specificity'] = specificity
    

    def accuracy(self):
        """Calculates the accuracy from the stored confusion matrix."""
        tp = self.matrix.iloc[1, 1]
        tn = self.matrix.iloc[0, 0]
        fp = self.matrix.iloc[0, 1]
        fn = self.matrix.iloc[1, 0]
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        self.results['accuracy'] = accuracy
    

    def error_rate(self):
        """Calculates the error rate from the stored confusion matrix."""
        tp = self.matrix.iloc[1, 1]
        tn = self.matrix.iloc[0, 0]
        fp = self.matrix.iloc[0, 1]
        fn = self.matrix.iloc[1, 0]
        error_rate = (fp + fn) / (tp + tn + fp + fn)
        self.results['error_rate'] = error_rate

    def overall_model_cost(self, matrix, cost_fp, cost_fn):
        """Calculates the overall model cost based on false positives and false negatives."""
        fp = matrix.iloc[0, 1]
        fn = matrix.iloc[1, 0]
        overall_cost = (fp * cost_fp) + (fn * cost_fn)
        self.results['overall_model_cost'] = overall_cost

    def get_results(self):
        """Returns all computed metric values."""
        self.error_rate()
        self.accuracy()
        self.sensitivity()
        self.specificity()
        for i in self.results:
            print(f"{i}: {self.results[i]:.4f}")




