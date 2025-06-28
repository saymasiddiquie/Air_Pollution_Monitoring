import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create synthetic dataset
np.random.seed(42)
n_samples = 1000

data = {
    'AOD': np.random.uniform(0.1, 1.5, n_samples),
    'Temperature': np.random.uniform(280, 320, n_samples),  
    'Humidity': np.random.uniform(30, 90, n_samples),      
    'WindSpeed': np.random.uniform(0.5, 5.0, n_samples),     
    'PBLH': np.random.uniform(500, 2000, n_samples),         
}

df = pd.DataFrame(data)

# Simulate PM2.5 using a formula + noise
df['PM2.5'] = (
    30 + 15 * df['AOD'] 
    - 0.05 * df['Temperature'] 
    + 0.1 * df['Humidity'] 
    - 2 * df['WindSpeed'] 
    - 0.005 * df['PBLH'] 
    + np.random.normal(0, 5, n_samples)
)

# Step 2: Define input/output
X = df[['AOD', 'Temperature', 'Humidity', 'WindSpeed', 'PBLH']]
y = df['PM2.5']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 6: Plot results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Observed PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Observed vs Predicted PM2.5")
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the trained model
import joblib
joblib.dump(model, 'model.pkl')
