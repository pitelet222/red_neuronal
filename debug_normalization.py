import numpy as np
import pandas as pd
import pickle
from red_neuronal import RedNeuronal
from utilities import normalize_data, divide_train_test

print("="*60)
print("DEBUG: DESNORMALIZACIÓN")
print("="*60)

# Cargar datos
df = pd.read_csv('data/sintetic_dataset.csv')
X = df.drop('Kwh_daily_consumption', axis=1).values
y = df['Kwh_daily_consumption'].values.reshape(-1, 1)

# Dividir
X_train, X_test, y_train, y_test = divide_train_test(X, y, test_ratio=0.2)

# Normalizar
X_train_norm, min_X, max_X = normalize_data(X_train)
X_test_norm, _, _ = normalize_data(X_test, min_X, max_X)
y_train_norm, min_y, max_y = normalize_data(y_train)
y_test_norm, _, _ = normalize_data(y_test, min_y, max_y)

print("\n1. PARÁMETROS DE NORMALIZACIÓN DE Y:")
print(f"   min_y shape: {min_y.shape}")
print(f"   max_y shape: {max_y.shape}")
print(f"   min_y value: {min_y}")
print(f"   max_y value: {max_y}")
print(f"   Rango: {max_y - min_y}")

print("\n2. VERIFICAR y_train_norm:")
print(f"   Shape: {y_train_norm.shape}")
print(f"   Min: {y_train_norm.min()}")
print(f"   Max: {y_train_norm.max()}")
print(f"   Mean: {y_train_norm.mean()}")

# Cargar modelo
with open('modelo_consumo.pkl', 'rb') as f:
    modelo = pickle.load(f)

red = RedNeuronal(9, 16, 8, 1)
red.W1, red.b1 = modelo['W1'], modelo['b1']
red.W2, red.b2 = modelo['W2'], modelo['b2']
red.W3, red.b3 = modelo['W3'], modelo['b3']

# Predicciones normalizadas
y_pred_norm = red.forward(X_test_norm)

print("\n3. PREDICCIONES NORMALIZADAS:")
print(f"   Shape: {y_pred_norm.shape}")
print(f"   Min: {y_pred_norm.min()}")
print(f"   Max: {y_pred_norm.max()}")
print(f"   Mean: {y_pred_norm.mean()}")
print(f"   Primeras 5 predicciones norm: {y_pred_norm[:5].flatten()}")

# Des-normalizar
print("\n4. DES-NORMALIZACIÓN:")
print(f"   Fórmula: y_pred = y_pred_norm * (max_y - min_y) + min_y")
print(f"   y_pred_norm * rango = {y_pred_norm[:5].flatten()} * {(max_y - min_y).flatten()}")

y_pred = y_pred_norm * (max_y - min_y) + min_y

print(f"\n   Resultado des-normalizado:")
print(f"   Min: {y_pred.min()}")
print(f"   Max: {y_pred.max()}")
print(f"   Mean: {y_pred.mean()}")
print(f"   Primeras 5 predicciones: {y_pred[:5].flatten()}")

print("\n5. COMPARAR CON VALORES REALES:")
print(f"   y_test shape: {y_test.shape}")
print(f"   y_test min: {y_test.min()}")
print(f"   y_test max: {y_test.max()}")
print(f"   y_test mean: {y_test.mean()}")
print(f"   Primeras 5 reales: {y_test[:5].flatten()}")

# Error
error = y_pred - y_test
print(f"\n6. ERROR:")
print(f"   Error medio: {np.mean(error)}")
print(f"   Error absoluto medio: {np.mean(np.abs(error))}")
print(f"   Primeros 5 errores: {error[:5].flatten()}")

# Calcular R² paso a paso
ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
r2 = 1 - (ss_res / ss_tot)

print(f"\n7. R² CÁLCULO:")
print(f"   SS_res (suma errores²): {ss_res}")
print(f"   SS_tot (suma (y-mean)²): {ss_tot}")
print(f"   R² = 1 - (SS_res/SS_tot) = {r2}")

# Test: ¿Qué pasa si uso la media como predicción?
y_pred_media = np.ones_like(y_test) * np.mean(y_test)
ss_res_media = np.sum((y_test - y_pred_media)**2)
r2_media = 1 - (ss_res_media / ss_tot)
print(f"\n8. COMPARACIÓN CON PREDICCIÓN INGENUA (usar media):")
print(f"   R² usando solo la media: {r2_media} (debería ser ~0)")
print(f"   Tu R²: {r2}")
print(f"   → Si tu R² < 0, tu modelo predice PEOR que la media")

# Verificar si el problema es broadcast
print(f"\n9. VERIFICAR DIMENSIONES EN OPERACIONES:")
print(f"   y_pred_norm.shape: {y_pred_norm.shape}")
print(f"   (max_y - min_y).shape: {(max_y - min_y).shape}")
print(f"   min_y.shape: {min_y.shape}")

# Test manual de des-normalización con un solo valor
print(f"\n10. TEST MANUAL DE DES-NORMALIZACIÓN:")
valor_norm_test = 0.5  # Punto medio
valor_real_test = valor_norm_test * (max_y - min_y) + min_y
print(f"   Si y_norm = 0.5 (punto medio)")
print(f"   Entonces y_real = 0.5 * {(max_y - min_y).flatten()[0]:.2f} + {min_y.flatten()[0]:.2f}")
print(f"             y_real = {valor_real_test.flatten()[0]:.2f} kWh")
print(f"   ¿Tiene sentido? Media real es {np.mean(y_test):.2f} kWh")

print("\n" + "="*60)