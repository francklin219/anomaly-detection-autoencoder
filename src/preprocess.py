import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")

TRAIN_TXT = os.path.join(RAW_DIR, "ECG5000_TRAIN.txt")
TEST_TXT = os.path.join(RAW_DIR, "ECG5000_TEST.txt")

X_TRAIN_NPY = os.path.join(OUT_DIR, "x_train.npy")
X_TEST_NPY = os.path.join(OUT_DIR, "x_test.npy")
Y_TEST_NPY = os.path.join(OUT_DIR, "y_test.npy")

def load_txt(path: str) -> pd.DataFrame:
    # Fichiers: label + 140 features (souvent séparés par espaces)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_df = load_txt(TRAIN_TXT)
    test_df = load_txt(TEST_TXT)

    # Première colonne = label
    y_train = train_df.iloc[:, 0].astype(int).to_numpy()
    x_train = train_df.iloc[:, 1:].to_numpy(dtype=np.float32)

    y_test = test_df.iloc[:, 0].astype(int).to_numpy()
    x_test = test_df.iloc[:, 1:].to_numpy(dtype=np.float32)


    x_train_normal = x_train[y_train == 1]

    scaler = StandardScaler()
    x_train_normal = scaler.fit_transform(x_train_normal)
    x_test_scaled = scaler.transform(x_test)

 
    np.save(os.path.join(OUT_DIR, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(OUT_DIR, "scaler_scale.npy"), scaler.scale_)

    np.save(X_TRAIN_NPY, x_train_normal)
    np.save(X_TEST_NPY, x_test_scaled)
    np.save(Y_TEST_NPY, y_test)

    print("Saved processed arrays:")
    print("x_train (normal):", x_train_normal.shape)
    print("x_test:", x_test_scaled.shape)
    print("y_test:", y_test.shape)
    print("Normal ratio in test:", (y_test == 1).mean())

if __name__ == "__main__":
    main()
