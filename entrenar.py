import numpy as np
import pandas as pd 
from red_neuronal import RedNeuronal
from utilities import (normalize_data, divide_train_test, save_model, metrics_calculator)
from visualizar import (plot_learning_curve, plot_predictions_vs_reality, plot_error_distribution)

def main():
    print("="*60)
    print("ENTRENAMIENTO DE RED NEURONAL - PREDICCIÓN CONSUMO ELÉCTRICO")
    print("="*60)
    
    # 1. CARGAR DATOS
    print("\n[1/6] Cargando datos...")
    df = pd.read_csv('data/sintetic_dataset.csv')
    print(f"✓ Datos cargados: {len(df)} muestras")
    print(f"✓ Estadísticas del consumo:")
    print(df['Kwh_daily_consumption'].describe())
    
    # 2. PREPARAR DATOS
    print("\n[2/6] Preparando datos...")
    
    # Separar features y target
    X = df.drop('Kwh_daily_consumption', axis=1).values
    y = df['Kwh_daily_consumption'].values.reshape(-1, 1)
    
    # Dividir train/test
    X_train, X_test, y_train, y_test = divide_train_test(
        X, y, test_ratio=0.2, shuffle=True
    )
    print(f"✓ Train: {len(X_train)} muestras")
    print(f"✓ Test: {len(X_test)} muestras")
    
    # Normalizar
    X_train_norm, min_vals_X, max_vals_X = normalize_data(X_train)
    X_test_norm, _, _ = normalize_data(X_test, min_vals_X, max_vals_X)
    
    y_train_norm, min_val_y, max_val_y = normalize_data(y_train)
    y_test_norm, _, _ = normalize_data(y_test, min_val_y, max_val_y)
    
    print("✓ Datos normalizados")
    
    # 3. CREAR RED
    print("\n[3/6] Creando red neuronal...")
    red = RedNeuronal(
        input_size=9,
        hidden1=16,
        hidden2=8,
        output_size=1
    )
    print("✓ Arquitectura: 9 → 16 → 8 → 1")
    
    # 4. ENTRENAR
    print("\n[4/6] Entrenando red...")
    print("Esto puede tardar 1-2 minutos...")
    
    history = red.entrenar(
        X_train_norm, y_train_norm,
        X_test_norm, y_test_norm,
        epochs=1500,
        batch_size=128,
        learning_rate=0.015,
        lambda_reg=0.01,
        verbose=True
    )
    
    print("\n✓ Entrenamiento completado")
    
    # 5. EVALUAR
    print("\n[5/6] Evaluando rendimiento...")
    
    # Predicciones en test
    y_pred_norm = red.predict(X_test_norm)
    
    
    # Des-normalizar para métricas interpretables
    from utilities import unnormalize_data
    y_pred = unnormalize_data(y_pred_norm, min_val_y, max_val_y)
    
    # Calcular métricas
    metricas = metrics_calculator(y_test, y_pred)
    
    print("\n📊 RESULTADOS EN TEST:")
    print(f"  • MSE:  {metricas['MSE']:.4f}")
    print(f"  • RMSE: {metricas['RMSE']:.4f} kWh")
    print(f"  • MAE:  {metricas['MAE']:.4f} kWh")
    print(f"  • R²:   {metricas['R2']:.4f}")
    
    if metricas['R2'] > 0.85:
        print("\n✓ ¡Excelente! R² > 0.85 indica buen ajuste")
    elif metricas['R2'] > 0.70:
        print("\n✓ Buen resultado. R² > 0.70")
    else:
        print("\n⚠ R² bajo. Considera ajustar hiperparámetros")
    
    # 6. VISUALIZAR
    print("\n[6/6] Generando visualizaciones...")
    
    plot_learning_curve(history, save=True)
    plot_predictions_vs_reality(y_test, y_pred, save=True)
    plot_error_distribution(y_test, y_pred, save=True)
    
    print("✓ Gráficas guardadas")
    
    # 7. GUARDAR MODELO
    print("\n[7/6] Guardando modelo...")
    save_model(red, min_vals_X, max_vals_X, 'modelo_consumo.pkl')
    
    # También guardar parámetros de normalización del target
    import pickle
    with open('normalizacion_y.pkl', 'wb') as f:
        pickle.dump({'min': min_val_y, 'max': max_val_y}, f)
    
    print("\n" + "="*60)
    print("✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    # 8. PRUEBA CON EJEMPLO
    print("\n🔍 Prueba con caso específico:")
    print("Casa: 4 personas, 120m², sin gas, con inducción")
    print("      con AC, sin calef. eléc., mes julio (verano)")
    
    caso_test = np.array([[4, 120, 0, 1, 0, 1, 0, 7, 3]])
    caso_norm, _, _ = normalize_data(caso_test, min_vals_X, max_vals_X)
    pred_norm = red.predict(caso_norm)
    
    # Justo antes de: pred_real = unnormalize_data(pred_norm, min_val_y, max_val_y)
    print(f"DEBUG CASO - pred_norm: {pred_norm}")
    print(f"DEBUG CASO - pred_norm shape: {pred_norm.shape}")
    print(f"DEBUG CASO - min_val_y: {min_val_y}")
    print(f"DEBUG CASO - max_val_y: {max_val_y}")
    print(f"DEBUG FINAL - pred_real calculado: {pred_norm[0,0] * (max_val_y[0] - min_val_y[0]) + min_val_y[0]:.2f} kWh/día")
    pred_real = unnormalize_data(pred_norm, min_val_y, max_val_y)
    
    print(f"→ Consumo predicho: {pred_real[0,0]:.2f} kWh/día")
    print(f"  (Aproximadamente {pred_real[0,0]*30:.0f} kWh/mes)")

if __name__ == "__main__":
    main()