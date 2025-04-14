from sklearn.ensemble import RandomForestClassifier
import joblib

# Cargar datos procesados
X_train, X_test, y_train, y_test = joblib.load('../processed_data.pkl')

# Entrenar modelo
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo
joblib.dump(model, '../model.pkl')