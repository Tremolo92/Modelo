import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Cargar datos
df = pd.read_csv("../data/ObesityDataSet.csv")

# Codificar variables categóricas
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
for col in categorical_cols:
    df[col] = label_encoders[col].transform(df[col])

# Guardar los codificadores para uso futuro
joblib.dump(label_encoders, '../label_encoders.pkl')

# Escalar y dividir datos
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar datos procesados
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), '../processed_data.pkl')