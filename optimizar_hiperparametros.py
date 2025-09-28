import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

def tanh_activation(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def optimizar_hiperparametros():
    """
    OPTIMIZACIÓN DE HIPERPARÁMETROS AVANZADA
    Prueba múltiples configuraciones para maximizar el rendimiento
    """
    print("🔬 OPTIMIZACIÓN DE HIPERPARÁMETROS - ANÁLISIS EXHAUSTIVO")
    print("="*70)
    
    try:
        # Cargar datos
        df = pd.read_csv('data/dataset_super_combinado.csv')
        X = df[['Avg VTAT', 'Avg CTAT', 'Booking Value', 'Ride Distance']].values
        Y = df[['is_completed']].values
        
        print(f"✅ Dataset cargado: {X.shape[0]:,} muestras")
        
        # Normalizar datos para mejor convergencia
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std
        
        print("✅ Datos normalizados (mean=0, std=1)")
        
        # Dividir datos: 80% entrenamiento, 20% validación
        split_idx = int(0.8 * len(X_norm))
        indices = np.random.permutation(len(X_norm))
        
        X_train = X_norm[indices[:split_idx]]
        Y_train = Y[indices[:split_idx]]
        X_val = X_norm[indices[split_idx:]]
        Y_val = Y[indices[split_idx:]]
        
        print(f"✅ División: {len(X_train):,} entrenamiento, {len(X_val):,} validación")
        
        # CONFIGURACIONES A PROBAR
        configuraciones = [
            # Config 1: Red original optimizada
            {
                'name': 'Red Original Optimizada',
                'architecture': [4, 6, 1],
                'learning_rate': 0.1,
                'epochs': 3000,
                'lambda_reg': 0.001,
                'activation': 'tanh'
            },
            # Config 2: Red expandida con mejor LR
            {
                'name': 'Red Expandida LR Alto',
                'architecture': [4, 8, 4, 1],
                'learning_rate': 0.05,
                'epochs': 4000,
                'lambda_reg': 0.01,
                'activation': 'leaky_relu'
            },
            # Config 3: Red simple con alta regularización
            {
                'name': 'Red Simple Alta Reg',
                'architecture': [4, 10, 1],
                'learning_rate': 0.2,
                'epochs': 2000,
                'lambda_reg': 0.001,
                'activation': 'tanh'
            },
            # Config 4: Red profunda pequeña
            {
                'name': 'Red Profunda Pequeña',
                'architecture': [4, 6, 3, 1],
                'learning_rate': 0.08,
                'epochs': 3500,
                'lambda_reg': 0.005,
                'activation': 'leaky_relu'
            },
            # Config 5: Red ancha
            {
                'name': 'Red Ancha',
                'architecture': [4, 12, 1],
                'learning_rate': 0.15,
                'epochs': 2500,
                'lambda_reg': 0.002,
                'activation': 'tanh'
            }
        ]
        
        resultados = []
        mejor_config = None
        mejor_accuracy = 0
        
        print(f"\n🧪 PROBANDO {len(configuraciones)} CONFIGURACIONES:")
        print("="*70)
        
        for i, config in enumerate(configuraciones):
            print(f"\n[{i+1}/{len(configuraciones)}] {config['name']}")
            print(f"   Arquitectura: {' → '.join(map(str, config['architecture']))}")
            print(f"   LR: {config['learning_rate']}, Épocas: {config['epochs']}")
            
            start_time = datetime.now()
            
            # Entrenar red con esta configuración
            modelo = entrenar_configuracion(
                X_train, Y_train, X_val, Y_val,
                config, X_mean, X_std
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            print(f"   ⏱️  Tiempo: {elapsed:.1f}s")
            print(f"   📈 Precisión entrenamiento: {modelo['train_accuracy']:.2f}%")
            print(f"   📊 Precisión validación: {modelo['val_accuracy']:.2f}%")
            
            # Guardar resultado
            resultado = {
                'config': config,
                'train_accuracy': modelo['train_accuracy'],
                'val_accuracy': modelo['val_accuracy'],
                'time': elapsed,
                'overfitting': modelo['train_accuracy'] - modelo['val_accuracy']
            }
            resultados.append(resultado)
            
            # Verificar si es el mejor
            if modelo['val_accuracy'] > mejor_accuracy:
                mejor_accuracy = modelo['val_accuracy']
                mejor_config = resultado
                
                # Guardar el mejor modelo
                guardar_mejor_modelo(modelo, config, X_mean, X_std)
        
        # ANÁLISIS DE RESULTADOS
        print("\n" + "="*70)
        print("📊 ANÁLISIS DE RESULTADOS")
        print("="*70)
        
        # Ordenar por precisión de validación
        resultados.sort(key=lambda x: x['val_accuracy'], reverse=True)
        
        print("\n🏆 RANKING DE CONFIGURACIONES:")
        for i, res in enumerate(resultados):
            config = res['config']
            overfitting = res['overfitting']
            overfitting_status = "⚠️ " if overfitting > 5 else "✅"
            
            print(f"\n[{i+1}] {config['name']}")
            print(f"    Validación: {res['val_accuracy']:.2f}% | "
                  f"Entrenamiento: {res['train_accuracy']:.2f}% | "
                  f"Overfitting: {overfitting:+.1f}% {overfitting_status}")
            print(f"    Tiempo: {res['time']:.1f}s | "
                  f"Arquitectura: {' → '.join(map(str, config['architecture']))}")
        
        # MEJOR CONFIGURACIÓN
        print(f"\n🥇 MEJOR CONFIGURACIÓN:")
        print(f"   {mejor_config['config']['name']}")
        print(f"   Precisión validación: {mejor_config['val_accuracy']:.2f}%")
        print(f"   Overfitting: {mejor_config['overfitting']:+.1f}%")
        print(f"   Arquitectura: {' → '.join(map(str, mejor_config['config']['architecture']))}")
        
        # RECOMENDACIONES
        print(f"\n💡 RECOMENDACIONES:")
        if mejor_config['val_accuracy'] > 95:
            print(f"   ✅ Excelente rendimiento alcanzado")
        elif mejor_config['val_accuracy'] > 90:
            print(f"   ⚡ Buen rendimiento, considerar más épocas")
        else:
            print(f"   🔧 Necesita más optimización")
            
        if mejor_config['overfitting'] > 10:
            print(f"   ⚠️  Alto overfitting, aumentar regularización")
        elif mejor_config['overfitting'] < 2:
            print(f"   📈 Bajo overfitting, puede aumentar capacidad")
        
        return mejor_config
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def entrenar_configuracion(X_train, Y_train, X_val, Y_val, config, X_mean, X_std):
    """Entrena una red con configuración específica"""
    
    # Inicialización He para LeakyReLU, Xavier para Tanh
    np.random.seed(42)
    
    if len(config['architecture']) == 3:  # Red de 2 capas
        n_x, n_h, n_y = config['architecture']
        
        if config['activation'] == 'leaky_relu':
            W1 = np.random.randn(n_x, n_h) * np.sqrt(2.0 / n_x)
            W2 = np.random.randn(n_h, n_y) * np.sqrt(2.0 / n_h)
        else:  # tanh
            W1 = np.random.randn(n_x, n_h) * np.sqrt(1.0 / n_x)
            W2 = np.random.randn(n_h, n_y) * np.sqrt(1.0 / n_h)
            
        b1 = np.zeros((1, n_h))
        b2 = np.zeros((1, n_y))
        
        # Función de activación
        if config['activation'] == 'leaky_relu':
            activation_fn = leaky_relu
            activation_deriv = leaky_relu_derivative
        else:
            activation_fn = tanh_activation
            activation_deriv = tanh_derivative
        
        # Entrenamiento
        lr = config['learning_rate']
        
        for epoch in range(config['epochs']):
            # Forward pass
            Z1 = X_train.dot(W1) + b1
            A1 = activation_fn(Z1)
            Z2 = A1.dot(W2) + b2
            A2 = sigmoid(Z2)
            
            # Loss con regularización
            m = Y_train.shape[0]
            A2_clipped = np.clip(A2, 1e-15, 1 - 1e-15)
            loss_main = -np.mean(Y_train * np.log(A2_clipped) + (1 - Y_train) * np.log(1 - A2_clipped))
            loss_reg = config['lambda_reg'] * (np.sum(W1**2) + np.sum(W2**2)) / (2 * m)
            
            # Backward pass
            dA2 = (A2 - Y_train) / m
            dZ2 = dA2 * sigmoid_derivative(A2)
            dW2 = A1.T.dot(dZ2) + config['lambda_reg'] * W2 / m
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            
            dA1 = dZ2.dot(W2.T)
            
            if config['activation'] == 'leaky_relu':
                dZ1 = dA1 * leaky_relu_derivative(Z1)
            else:
                dZ1 = dA1 * activation_deriv(Z1)
                
            dW1 = X_train.T.dot(dZ1) + config['lambda_reg'] * W1 / m
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            # Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            
            # Learning rate decay
            if epoch > 0 and epoch % 1000 == 0:
                lr *= 0.95
        
        # Evaluación final
        # Entrenamiento
        Z1_train = X_train.dot(W1) + b1
        A1_train = activation_fn(Z1_train)
        Z2_train = A1_train.dot(W2) + b2
        A2_train = sigmoid(Z2_train)
        train_pred = (A2_train > 0.5).astype(int)
        train_accuracy = np.mean(train_pred == Y_train) * 100
        
        # Validación
        Z1_val = X_val.dot(W1) + b1
        A1_val = activation_fn(Z1_val)
        Z2_val = A1_val.dot(W2) + b2
        A2_val = sigmoid(Z2_val)
        val_pred = (A2_val > 0.5).astype(int)
        val_accuracy = np.mean(val_pred == Y_val) * 100
        
        return {
            'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'is_3_layer': False
        }
        
    else:  # Red de 3 capas
        n_x, n_h1, n_h2, n_y = config['architecture']
        
        if config['activation'] == 'leaky_relu':
            W1 = np.random.randn(n_x, n_h1) * np.sqrt(2.0 / n_x)
            W2 = np.random.randn(n_h1, n_h2) * np.sqrt(2.0 / n_h1)
            W3 = np.random.randn(n_h2, n_y) * np.sqrt(2.0 / n_h2)
        else:
            W1 = np.random.randn(n_x, n_h1) * np.sqrt(1.0 / n_x)
            W2 = np.random.randn(n_h1, n_h2) * np.sqrt(1.0 / n_h1)
            W3 = np.random.randn(n_h2, n_y) * np.sqrt(1.0 / n_h2)
            
        b1 = np.zeros((1, n_h1))
        b2 = np.zeros((1, n_h2))
        b3 = np.zeros((1, n_y))
        
        if config['activation'] == 'leaky_relu':
            activation_fn = leaky_relu
            activation_deriv = leaky_relu_derivative
        else:
            activation_fn = tanh_activation
            activation_deriv = tanh_derivative
        
        lr = config['learning_rate']
        
        for epoch in range(config['epochs']):
            # Forward pass
            Z1 = X_train.dot(W1) + b1
            A1 = activation_fn(Z1)
            Z2 = A1.dot(W2) + b2
            A2 = activation_fn(Z2)
            Z3 = A2.dot(W3) + b3
            A3 = sigmoid(Z3)
            
            # Loss con regularización
            m = Y_train.shape[0]
            A3_clipped = np.clip(A3, 1e-15, 1 - 1e-15)
            loss_main = -np.mean(Y_train * np.log(A3_clipped) + (1 - Y_train) * np.log(1 - A3_clipped))
            loss_reg = config['lambda_reg'] * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)) / (2 * m)
            
            # Backward pass
            dA3 = (A3 - Y_train) / m
            dZ3 = dA3 * sigmoid_derivative(A3)
            dW3 = A2.T.dot(dZ3) + config['lambda_reg'] * W3 / m
            db3 = np.sum(dZ3, axis=0, keepdims=True)
            
            dA2 = dZ3.dot(W3.T)
            dZ2 = dA2 * activation_deriv(Z2)
            dW2 = A1.T.dot(dZ2) + config['lambda_reg'] * W2 / m
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            
            dA1 = dZ2.dot(W2.T)
            dZ1 = dA1 * activation_deriv(Z1)
            dW1 = X_train.T.dot(dZ1) + config['lambda_reg'] * W1 / m
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            # Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            W3 -= lr * dW3
            b3 -= lr * db3
            
            # Learning rate decay
            if epoch > 0 and epoch % 1000 == 0:
                lr *= 0.95
        
        # Evaluación final
        # Entrenamiento
        Z1_train = X_train.dot(W1) + b1
        A1_train = activation_fn(Z1_train)
        Z2_train = A1_train.dot(W2) + b2
        A2_train = activation_fn(Z2_train)
        Z3_train = A2_train.dot(W3) + b3
        A3_train = sigmoid(Z3_train)
        train_pred = (A3_train > 0.5).astype(int)
        train_accuracy = np.mean(train_pred == Y_train) * 100
        
        # Validación
        Z1_val = X_val.dot(W1) + b1
        A1_val = activation_fn(Z1_val)
        Z2_val = A1_val.dot(W2) + b2
        A2_val = activation_fn(Z2_val)
        Z3_val = A2_val.dot(W3) + b3
        A3_val = sigmoid(Z3_val)
        val_pred = (A3_val > 0.5).astype(int)
        val_accuracy = np.mean(val_pred == Y_val) * 100
        
        return {
            'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'is_3_layer': True
        }

def guardar_mejor_modelo(modelo, config, X_mean, X_std):
    """Guarda el mejor modelo encontrado"""
    filename = 'modelo_optimizado.npz'
    
    if modelo['is_3_layer']:
        np.savez(filename,
                 W1=modelo['W1'], b1=modelo['b1'],
                 W2=modelo['W2'], b2=modelo['b2'],
                 W3=modelo['W3'], b3=modelo['b3'],
                 train_accuracy=modelo['train_accuracy'],
                 val_accuracy=modelo['val_accuracy'],
                 X_mean=X_mean, X_std=X_std,
                 config_name=config['name'],
                 architecture=config['architecture'],
                 is_3_layer=True)
    else:
        np.savez(filename,
                 W1=modelo['W1'], b1=modelo['b1'],
                 W2=modelo['W2'], b2=modelo['b2'],
                 train_accuracy=modelo['train_accuracy'],
                 val_accuracy=modelo['val_accuracy'],
                 X_mean=X_mean, X_std=X_std,
                 config_name=config['name'],
                 architecture=config['architecture'],
                 is_3_layer=False)
    
    print(f"   💾 Modelo guardado: {filename}")

def crear_predictor_optimizado(archivo='modelo_optimizado.npz'):
    """Crea predictor con el modelo optimizado"""
    try:
        modelo = np.load(archivo, allow_pickle=True)
        
        W1, b1 = modelo['W1'], modelo['b1']
        W2, b2 = modelo['W2'], modelo['b2']
        X_mean, X_std = modelo['X_mean'], modelo['X_std']
        is_3_layer = modelo['is_3_layer']
        config_name = str(modelo['config_name'])
        
        if is_3_layer:
            W3, b3 = modelo['W3'], modelo['b3']
        
        def predecir_optimizado(vtat, ctat, valor, distancia):
            """Predice con el modelo optimizado"""
            X_new = np.array([[vtat, ctat, valor, distancia]])
            X_new_norm = (X_new - X_mean) / X_std
            
            # Forward pass
            Z1 = X_new_norm.dot(W1) + b1
            
            # Determinar función de activación por nombre de config
            if 'LeakyReLU' in config_name or 'Expandida' in config_name:
                A1 = leaky_relu(Z1)
            else:
                A1 = tanh_activation(Z1)
            
            Z2 = A1.dot(W2) + b2
            
            if is_3_layer:
                if 'LeakyReLU' in config_name or 'Expandida' in config_name:
                    A2 = leaky_relu(Z2)
                else:
                    A2 = tanh_activation(Z2)
                Z3 = A2.dot(W3) + b3
                A3 = sigmoid(Z3)
                return A3[0, 0] * 100
            else:
                A2 = sigmoid(Z2)
                return A2[0, 0] * 100
        
        print(f"🔮 PREDICTOR OPTIMIZADO LISTO")
        print(f"   Modelo: {config_name}")
        print(f"   Precisión validación: {modelo['val_accuracy']:.2f}%")
        
        return predecir_optimizado
        
    except FileNotFoundError:
        print(f"❌ No se encontró {archivo}")
        return None

if __name__ == "__main__":
    # Ejecutar optimización
    mejor = optimizar_hiperparametros()
    
    if mejor:
        print(f"\n🎯 OPTIMIZACIÓN COMPLETADA")
        print(f"🏆 Mejor precisión: {mejor['val_accuracy']:.2f}%")
        
        # Crear predictor optimizado
        predictor = crear_predictor_optimizado()
        
        if predictor:
            print(f"\n🧪 PRUEBA DEL PREDICTOR:")
            # Casos de prueba
            casos = [
                (8, 20, 300, 12, "Viaje típico urbano"),
                (15, 45, 150, 5, "Viaje corto alto tiempo"),
                (5, 10, 500, 25, "Viaje largo eficiente"),
                (25, 60, 100, 8, "Viaje problemático")
            ]
            
            for vtat, ctat, valor, dist, desc in casos:
                prob = predictor(vtat, ctat, valor, dist)
                print(f"   {desc}: {prob:.1f}% completación")
    else:
        print(f"\n❌ Error en la optimización")