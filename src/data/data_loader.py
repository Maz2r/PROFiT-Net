import numpy as np

class DataLoader:
    @staticmethod
    def load_data(train_path, val_path, train_label_path, val_label_path):
        X_train = np.load(train_path)
        X_val = np.load(val_path)
        y_train = np.load(train_label_path)
        y_val = np.load(val_label_path)
        return X_train, X_val, y_train, y_val
