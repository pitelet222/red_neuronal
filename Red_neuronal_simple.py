import numpy as np
import matplotlib.pyplot as plt

# ---------- FUNCIONES DE ACTIVACIÓN ----------
def sigmoid(z):
    """
    Función sigmoid con protección contra overflow
    """
    # Clip z para evitar overflow en exp(-z)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """
    Derivada de sigmoid donde a = sigmoid(z)
    """
    return a * (1 - a)

def tanh_derivative(a):
    """
    Derivada de tanh donde a = tanh(z)
    """
    return 1 - a**2

def relu(z):
    """
    Función ReLU (alternativa a tanh)
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    Derivada de ReLU
    """
    return (z > 0).astype(float)

# ---------- INICIALIZACIÓN DE PARÁMETROS ----------
def init_params(n_x, n_h, n_y, initialization='xavier', seed=42):
    """
    Inicializa los parámetros de la red neuronal
    
    Args:
        n_x: número de características de entrada
        n_h: número de neuronas en la capa oculta
        n_y: número de neuronas de salida
        initialization: tipo de inicialización ('xavier', 'he', 'random')
        seed: semilla para reproducibilidad
    """
    np.random.seed(seed)
    
    if initialization == 'xavier':
        # Xavier/Glorot initialization - buena para sigmoid/tanh
        W1 = np.random.randn(n_x, n_h) * np.sqrt(1.0 / n_x)
        W2 = np.random.randn(n_h, n_y) * np.sqrt(1.0 / n_h)
    elif initialization == 'he':
        # He initialization - mejor para ReLU
        W1 = np.random.randn(n_x, n_h) * np.sqrt(2.0 / n_x)
        W2 = np.random.randn(n_h, n_y) * np.sqrt(2.0 / n_h)
    else:
        # Inicialización aleatoria simple
        W1 = np.random.randn(n_x, n_h) * 0.1
        W2 = np.random.randn(n_h, n_y) * 0.1
    
    b1 = np.zeros((1, n_h))
    b2 = np.zeros((1, n_y))
    
    return W1, b1, W2, b2

# ---------- PROPAGACIÓN HACIA ADELANTE ----------
def forward(X, W1, b1, W2, b2, activation='tanh'):
    """
    Realiza la propagación hacia adelante
    
    Args:
        X: datos de entrada (m, n_x)
        W1, b1, W2, b2: parámetros de la red
        activation: función de activación para la capa oculta ('tanh' o 'relu')
    
    Returns:
        A2: salida de la red
        cache: valores intermedios para backpropagation
    """
    # Capa oculta
    Z1 = X.dot(W1) + b1
    
    if activation == 'tanh':
        A1 = np.tanh(Z1)
    elif activation == 'relu':
        A1 = relu(Z1)
    else:
        raise ValueError("Activación no soportada")
    
    # Capa de salida
    Z2 = A1.dot(W2) + b2
    A2 = sigmoid(Z2)
    
    cache = (X, Z1, A1, Z2, A2, activation)
    return A2, cache

# ---------- FUNCIÓN DE PÉRDIDA ----------
def compute_loss(A2, Y, loss_type='mse'):
    """
    Calcula la pérdida
    
    Args:
        A2: predicciones (m, n_y)
        Y: etiquetas verdaderas (m, n_y)
        loss_type: tipo de pérdida ('mse' o 'binary_crossentropy')
    """
    m = Y.shape[0]
    
    if loss_type == 'mse':
        loss = np.mean((A2 - Y) ** 2)
    elif loss_type == 'binary_crossentropy':
        # Evitar log(0) con clipping
        A2_clipped = np.clip(A2, 1e-15, 1 - 1e-15)
        loss = -np.mean(Y * np.log(A2_clipped) + (1 - Y) * np.log(1 - A2_clipped))
    else:
        raise ValueError("Tipo de pérdida no soportado")
    
    return loss

# ---------- PROPAGACIÓN HACIA ATRÁS ----------
def backward(cache, W2, A2, Y, loss_type='mse'):
    """
    Realiza la propagación hacia atrás (backpropagation)
    """
    X, Z1, A1, Z2, A2_cache, activation = cache
    m = X.shape[0]
    
    # Gradientes de la capa de salida
    if loss_type == 'mse':
        dA2 = 2 * (A2 - Y) / m
    elif loss_type == 'binary_crossentropy':
        dA2 = (A2 - Y) / m
    else:
        raise ValueError("Tipo de pérdida no soportado")
    
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = A1.T.dot(dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # Gradientes de la capa oculta
    dA1 = dZ2.dot(W2.T)
    
    if activation == 'tanh':
        dZ1 = dA1 * tanh_derivative(A1)
    elif activation == 'relu':
        dZ1 = dA1 * relu_derivative(Z1)
    
    dW1 = X.T.dot(dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2

# ---------- ACTUALIZACIÓN DE PARÁMETROS ----------
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    """
    Actualiza los parámetros usando gradiente descendente
    """
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    
    return W1, b1, W2, b2

# ---------- ENTRENAMIENTO ----------
def train(X, Y, n_h=4, epochs=10000, lr=0.5, activation='tanh', 
          loss_type='binary_crossentropy', print_every=1000, plot_loss=True):
    """
    Entrena la red neuronal
    
    Args:
        X: datos de entrada (m, n_x)
        Y: etiquetas (m, n_y)
        n_h: número de neuronas en capa oculta
        epochs: número de épocas
        lr: tasa de aprendizaje
        activation: función de activación ('tanh' o 'relu')
        loss_type: tipo de pérdida ('mse' o 'binary_crossentropy')
        print_every: frecuencia de impresión
        plot_loss: si graficar la pérdida
    """
    n_x = X.shape[1]
    n_y = Y.shape[1]
    
    # Inicializar parámetros
    init_type = 'he' if activation == 'relu' else 'xavier'
    W1, b1, W2, b2 = init_params(n_x, n_h, n_y, initialization=init_type)
    
    losses = []
    
    print(f"Iniciando entrenamiento...")
    print(f"Arquitectura: {n_x} -> {n_h} -> {n_y}")
    print(f"Activación: {activation}, Pérdida: {loss_type}")
    print(f"Tasa de aprendizaje: {lr}")
    print("-" * 50)
    
    for i in range(epochs):
        # Forward pass
        A2, cache = forward(X, W1, b1, W2, b2, activation)
        
        # Calcular pérdida
        loss = compute_loss(A2, Y, loss_type)
        losses.append(loss)
        
        # Backward pass
        dW1, db1, dW2, db2 = backward(cache, W2, A2, Y, loss_type)
        
        # Actualizar parámetros
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)
        
        # Imprimir progreso
        if print_every and i % print_every == 0:
            accuracy = compute_accuracy(A2, Y)
            print(f"Época {i:5d} - Pérdida: {loss:.6f} - Precisión: {accuracy:.2f}%")
    
    # Gráfico de pérdida
    if plot_loss:
        plot_training_loss(losses)
    
    print("-" * 50)
    print("Entrenamiento completado!")
    
    return W1, b1, W2, b2, losses

# ---------- PREDICCIÓN ----------
def predict(X, W1, b1, W2, b2, activation='tanh', threshold=0.5):
    """
    Realiza predicciones con la red entrenada
    """
    A2, _ = forward(X, W1, b1, W2, b2, activation)
    predictions = (A2 > threshold).astype(int)
    return predictions, A2

# ---------- MÉTRICAS ----------
def compute_accuracy(A2, Y, threshold=0.5):
    """
    Calcula la precisión de las predicciones
    """
    predictions = (A2 > threshold).astype(int)
    accuracy = np.mean(predictions == Y) * 100
    return accuracy

# ---------- VISUALIZACIÓN ----------
def plot_training_loss(losses):
    """
    Grafica la evolución de la pérdida durante el entrenamiento
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Evolución de la Pérdida Durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True)
    plt.show()

def plot_decision_boundary(X, Y, W1, b1, W2, b2, activation='tanh'):
    """
    Visualiza la frontera de decisión (solo para datos 2D)
    """
    if X.shape[1] != 2:
        print("La visualización de frontera de decisión solo funciona para datos 2D")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Crear una malla de puntos
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Hacer predicciones en la malla
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    _, Z = predict(mesh_points, W1, b1, W2, b2, activation)
    Z = Z.reshape(xx.shape)
    
    # Graficar la frontera de decisión
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Probabilidad')
    
    # Graficar los puntos de datos
    colors = ['red', 'blue']
    for i in range(2):
        mask = (Y.ravel() == i)
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                   s=100, alpha=0.9, edgecolors='black',
                   label=f'Clase {i}')
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Frontera de Decisión de la Red Neuronal')
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------- EJEMPLO DE USO ----------
if __name__ == "__main__":
    # Datos XOR
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    
    print("=== ENTRENAMIENTO DE RED NEURONAL PARA XOR ===\n")
    
    # Entrenar la red
    W1, b1, W2, b2, losses = train(
        X, Y, 
        n_h=4,                           # Neuronas en capa oculta
        epochs=5000,                     # Número de épocas
        lr=0.5,                          # Tasa de aprendizaje
        activation='tanh',               # Función de activación
        loss_type='binary_crossentropy', # Tipo de pérdida
        print_every=1000                 # Frecuencia de impresión
    )
    
    # Hacer predicciones
    predictions, probabilities = predict(X, W1, b1, W2, b2, 'tanh')
    
    print("\n=== RESULTADOS FINALES ===")
    print("Entradas:")
    print(X)
    print("\nSalidas esperadas:")
    print(Y.ravel())
    print("\nPredicciones:")
    print(predictions.ravel())
    print("\nProbabilidades:")
    print(np.round(probabilities.ravel(), 4))
    
    # Calcular precisión final
    final_accuracy = compute_accuracy(probabilities, Y)
    print(f"\nPrecisión final: {final_accuracy:.2f}%")
    
    # Visualizar frontera de decisión
    plot_decision_boundary(X, Y, W1, b1, W2, b2, 'tanh')
