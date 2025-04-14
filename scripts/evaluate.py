from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cargar modelo y datos
model = joblib.load('../model.pkl')
_, X_test, _, y_test = joblib.load('../processed_data.pkl')

# Evaluar
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Matriz de confusión
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()