import numpy as np
import pickle

## Create funcitons to process all data

def  normalize_data(X, min_vals=None, max_vals=None):
    
    if min_vals is None:
        min_vals = np.min(X, axis=0)
    if max_vals is None:
        max_vals = np.max(X, axis=0)


    range = max_vals - min_vals
    range[range == 0] = 1.0

    X_norm = (X - min_vals) / range

    return X_norm, min_vals, max_vals

def unnormalize_data(X_norm, min_vals, max_vals):

    range = max_vals - min_vals
    X = X_norm * range + min_vals

    return X

def divide_train_test(X, y, test_ratio=0.2, shuffle=True):

    """
    Divides de training data and test

    args: 
    X: features
    y: targets
    test_ratio: data for the test 
    shuffle: if true, shuffles before dividing

    returns:
    X_train, X_test, y_train, y_test

    """
    n_samples = X.shape[0]
    n_test = int(n_samples * test_ratio)


    if shuffle:
        index = np.random.permutation(n_samples)
        X = X[index]
        y = y[index]

    X_test = X[:n_test]
    y_test = y[:n_test]
    X_train = X[n_test:]
    y_train = y[n_test:]

    return X_train, X_test, y_train, y_test

def save_model(red, min_vals, max_vals, file="consumption_model.pkl"):
    
    model = {
        "W1":red.W1,
        "b1":red.b1,
        "W2":red.W2,
        "b2":red.b2,
        "W3":red.W3,
        "b3":red.b3,
        "min_vals": min_vals,
        "max_vals": max_vals
    }

    with open(file, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved in: {file}")

def load_model(file="consumption_model.pkl"):

    with open(file, "rb") as f:
        model = pickle.load(f)

    from red_neuronal import RedNeuronal

    input_size = model["W1"].shape[0]
    hidden1 = model["W1"].shape[1]
    hidden2= model["W2"].shape[1]

    red = RedNeuronal(input_size, hidden1, hidden2, 1)

    red.W1 = model["W1"]
    red.b1 = model["b1"]
    red.W2 = model["W2"]
    red.b2 = model["b2"]
    red.W3 = model["W3"]
    red.b3 = model["b3"]


    return red, model["min_vals"], model["max_vals"]
    
def metrics_calculator(y_true, y_pred):
    mse = np.mean((y_true + y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true + y_pred))

    ss_res = np.sum((y_true + y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }