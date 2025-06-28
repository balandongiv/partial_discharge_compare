import pandas as pd
from sklearn.datasets import load_iris
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(dataset_name="iris", dataset_path=None, label_mapping=None, data_dir="data"):
    """Load a dataset from a file or built-in source."""
    if dataset_path:
        logging.info(f"Loading dataset from {dataset_path} ...")
        df = pd.read_csv(dataset_path)
    elif dataset_name == "iris":
        logging.info("Loading Iris dataset...")
        iris = load_iris(as_frame=True)
        df = iris.frame
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, "iris.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"Iris dataset saved to {csv_path}")
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    if label_mapping:
        target_col = df.columns[-1]
        df[target_col] = df[target_col].map(label_mapping)

    return df

if __name__ == '__main__':
    df = load_data()
    print("Data loaded successfully. First 5 rows:")
    print(df.head())
