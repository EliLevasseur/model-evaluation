"""Module specified for evaluating Model Performance."""

class Model:
    def __init__(self, matrix, model="Decision Tree"):
        """Initializes the Model with a confusion matrix, optional name,"""
        self.model = model
        self.matrix = matrix
        self.results = {}

    def _extract_matrix_values(self):
        """Extract confusion matrix values: tn, fp, fn, tp."""
        tn = self.matrix.iloc[0, 0]
        fp = self.matrix.iloc[0, 1]
        fn = self.matrix.iloc[1, 0]
        tp = self.matrix.iloc[1, 1]
        return tn, fp, fn, tp

    def _get_total_samples(self):
        """Calculate total number of samples."""
        tn, fp, fn, tp = self._extract_matrix_values()
        return tn + fp + fn + tp

    def __repr__(self):
        return f"This object can perform a series of operations to evaluate a models performance"

    def __str__(self):
        return f"{self.model=}"

    def sensitivity(self):
        """Calculates the sensitivity (true positive rate) from the stored confusion matrix."""
        _, _, fn, tp = self._extract_matrix_values()
        self.results['sensitivity'] = tp / (tp + fn)


    def specificity(self):
        """Calculates the specificity (true negative rate) from the stored confusion matrix."""
        tn, fp, _, _ = self._extract_matrix_values()
        self.results['specificity'] = tn / (tn + fp)
    

    def accuracy(self):
        """Calculates the accuracy from the stored confusion matrix."""
        tn, fp, fn, tp = self._extract_matrix_values()
        self.results['accuracy'] = (tp + tn) / self._get_total_samples()
    

    def error_rate(self):
        """Calculates the error rate from the stored confusion matrix."""
        if 'accuracy' not in self.results:
            self.accuracy()
        self.results['error_rate'] = 1 - self.results['accuracy']

    def overall_model_cost(self, cost_fp, cost_fn):
        """Calculates the overall model cost based on false positives and false negatives."""
        _, fp, fn, _ = self._extract_matrix_values()
        self.results['overall_model_cost'] = (fp * cost_fp) + (fn * cost_fn)

    def get_results(self):
        """Returns all computed metric values."""
        self.error_rate()
        self.accuracy()
        self.sensitivity()
        self.specificity()
        for i in self.results:
            print(f"{i}: {self.results[i]:.4f}")




