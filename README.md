# Model Evaluation Utility

This small Python module provides a `Model` class that calculates
common performance metrics (sensitivity, specificity, accuracy, error rate)
from a confusion matrix (as a pandas `DataFrame`).

## Usage

1. Create a virtual environment (the example uses `env`):

```bash
python -m venv env
source env/bin/activate        # or ``env\Scripts\activate`` on Windows
pip install pandas
#(optional) if you want to test the code
pip install -r requirements.txt
```

2. Import and use the class in your code:

```python
import pandas as pd
from skeval import Model

matrix = pd.DataFrame([[50, 10], [5, 35]])  # [[tn, fp], [fn, tp]]
model = Model(matrix, model="DecisionTree")
model.get_results()  # prints metrics
```

3. Run a test case (use python3 if python doesnt work)

```
python -m pytest test_skeval.py -q
```

