import numpy as np


class RedNeuronal:
    def __init__(self, input_size=9, hidden1=16, hidden2=8, output_size=1):

        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros((1,hidden1))

        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0/hidden1)
        self.b2 = np.zeros((1,hidden2))
        
        self.W3 = np.random.randn(hidden2, output_size) * np.sqrt(2.0/hidden2)
        self.b3 = np.zeros((1,output_size))

        ## Saves the data during the forward

        self.cache={}


    def relu(self, Z):
        return np.maximum(0,Z)
    
    def der_relu(self, Z):
        return (Z > 0).astype(float)
    
    def forward(self, X):

        #First layer

        Z1 = X @ self.W1 + self.b1  
        A1 = self.relu(Z1)

        #Second layer

        Z2 = A1 @ self.W2 + self.b2
        A2 = self.relu(Z2)

        #Third layer (output)

        Z3 = A2 @ self.W3 + self.b3
        output = Z3 

        self.cache = {
            "X":X,
            "Z1":Z1, "A1":A1,
            "Z2":Z2, "A2":A2,
            "Z3":Z3, "output": output
        }

        return output
    
    def calculate_loss(self,y_pred,y_true):
        
        n = y_true.shape[0]
        loss = np.sum((y_pred - y_true)**2) / n
        return loss
    

    def backward(self, y_true,learning_rate=0.015, lambda_reg=0):
        
        n = y_true.shape[0]

        X = self.cache["X"]
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]
        Z1 = self.cache["Z1"]
        Z2 = self.cache["Z2"]
        output = self.cache["output"]

        ## Gradient output

        dZ3 = (output - y_true) / n
        
        #Gradients L2 and W3

        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        #Reg for 

        if lambda_reg > 0:
            dW3 += (lambda_reg / n) * self.W3

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.der_relu(Z2)

        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        if lambda_reg > 0:
            dW2 += (lambda_reg / n) * self.W2

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.der_relu(Z1)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        if lambda_reg > 0:
            dW1 += (lambda_reg/n) * self.W1

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3


    def entrenar(self, X_train, y_train, X_val, y_val, epochs=1000, batch_size=32, learning_rate=0.015, lambda_reg=0, verbose=True):

        history = {
            "train_loss": [],
            "val_loss": []
        }

        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            index = np.random.permutation(n_samples)
            X_shuffled = X_train[index]
            y_shuffled = y_train[index]


            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]


                y_pred = self.forward(X_batch)
                self.backward(y_batch, learning_rate, lambda_reg)
            
            y_train_pred = self.forward(X_train)
            train_loss = self.calculate_loss(y_train_pred, y_train)

            y_val_pred = self.forward(X_val)
            val_loss = self.calculate_loss(y_val_pred, y_val)  # Asegúrate de usar y_val aquí

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)



            if verbose and epoch % 100 == 0:
                print(f"{epoch}/{epochs} = "
                      f"Train Loss: {train_loss:.4f} = "
                      f"Val loss: {val_loss:.4f}")
                
        return history
        
    def predict(self, X):
        return self.forward(X)
    

