New 
# Online News Popularity Prediction Pipeline

A modular machine learning regression pipeline built in Python to predict the number of social media shares for online news articles using Scikit-Learn.

## Project Structure

```root
├── data/                     # Dataset directory (configured via src/config.yaml)
├── eda.ipynb                 # Exploratory Data Analysis notebook
├── src/
│   ├── config.yaml           # Pipeline configuration (paths, target, feature lists)
│   ├── datapreparation_2.py  # Feature preprocessing pipelines (Preprocessor)
│   └── datatraining_1.py     # Data splitting, model training, tuning & evaluation (ModelTraining)
├── main12.py                 # Main entry point orchestrating the end-to-end pipeline
└── requirements.txt          # Python package dependencies
```

## Features & Workflow

1. **Configuration Management (`src/config.yaml`)**: Centralizes dataset file paths, target variable (`shares`), columns to drop (`ID`, `URL`), and feature groupings (numerical, nominal, and ordinal).
2. **Data Preprocessing (`src/datapreparation_2.py`)**: Uses a custom `Preprocessor` class to handle numerical, nominal categorical, and ordinal features dynamically based on the active dataset columns.
3. **Data Splitting (`src/datatraining_1.py`)**: Splits the dataset into training, validation, and test sets (`x_train`, `x_val`, `x_test`, `y_train`, `y_val`, `y_test`).
4. **Baseline & Tuned Model Training**:
   - Evaluates baseline regression models on the validation set.
   - Performs hyperparameter tuning (`ridge_tuned` and `lasso_tuned` across multiple `alpha` regularization values).
5. **Model Selection & Final Evaluation**: Automatically selects the top-performing model based on validation **$R^2$** score and evaluates it on the unseen test set.

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <https://github.com/mangasiao-ui/new>
   cd <new>
   ```

2. **Create and activate a virtual environment (optional)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Verify that `data_path` and feature lists in `src/config.yaml` match your local dataset location.
2. Run the main pipeline script from the project root directory:
   ```bash
   python main12.py
   ```

## Output

Running `main12.py` logs progress to the console and prints:
- Confirmation of data loading, preprocessing, and train/validation/test splitting
- The name of the **best-performing model** selected via validation $R^2$
- **Final test metrics** evaluated on the hold-out test set
