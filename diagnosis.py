import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("DIAGNÓSTICO DEL DATASET")
print("="*60)

# Cargar datos
df = pd.read_csv('data/sintetic_dataset.csv')

print("\n1. DISTRIBUCIÓN DE CONSUMO:")
print(df['Kwh_daily_consumption'].describe())

# Ver cuántos valores son exactamente 2.0
valores_minimos = (df['Kwh_daily_consumption'] == 2.0).sum()
print(f"\n⚠ Valores exactamente = 2.0: {valores_minimos} ({valores_minimos/len(df)*100:.1f}%)")

# Histograma
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['Kwh_daily_consumption'], bins=50, edgecolor='black')
plt.xlabel('Consumo (kWh/día)')
plt.ylabel('Frecuencia')
plt.title('Distribución del Consumo')
plt.axvline(df['Kwh_daily_consumption'].mean(), color='r', linestyle='--', label=f'Media: {df["Kwh_daily_consumption"].mean():.2f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot(df['Kwh_daily_consumption'])
plt.ylabel('Consumo (kWh/día)')
plt.title('Boxplot - Detección de Outliers')

plt.tight_layout()
plt.savefig('diagnostico_datos.png', dpi=150)
plt.show()

print("\n2. CORRELACIONES CON CONSUMO:")
correlaciones = df.corr()['Kwh_daily_consumption'].sort_values(ascending=False)
print(correlaciones)

print("\n3. VALORES ÚNICOS POR COLUMNA:")
for col in df.columns:
    if col != 'Kwh_daily_consumption':
        unicos = df[col].nunique()
        print(f"  {col}: {unicos} valores únicos")

print("\n✓ Diagnóstico completado. Revisa 'diagnostico_datos.png'")