import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data(df):
    """Preprocesa los datos para el modelo"""
    # Seleccionar características relevantes
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    # Crear una copia para no modificar el original
    processed = df[features].copy()
    
    # Convertir variables categóricas a numéricas
    label_encoders = {}
    for col in ['Sex', 'Embarked']:
        le = LabelEncoder()
        processed[col] = le.fit_transform(processed[col].fillna('Unknown'))
        label_encoders[col] = le
    
    # Manejar valores faltantes
    processed['Age'] = processed['Age'].fillna(processed['Age'].median())
    processed['Fare'] = processed['Fare'].fillna(processed['Fare'].median())
    processed['Embarked'] = processed['Embarked'].fillna(processed['Embarked'].mode()[0])
    
    return processed, label_encoders

def train_model(train_file):
    """Entrena el modelo Random Forest"""
    # Cargar datos de entrenamiento
    train_data = pd.read_csv(train_file)
    
    # Preprocesar datos
    X, label_encoders = preprocess_data(train_data)
    y = train_data['Survived']
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Crear y entrenar el modelo
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    print(f"Precisión en entrenamiento: {accuracy_score(y_train, train_pred):.2f}")
    print(f"Precisión en prueba: {accuracy_score(y_test, test_pred):.2f}")
    
    # Guardar modelo y encoders para uso futuro
    joblib.dump(model, 'titanic_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    return model, label_encoders

def predict_survival(predict_file, model, label_encoders):
    """Realiza predicciones sobre nuevos datos"""
    # Cargar datos a predecir
    predict_data = pd.read_csv(predict_file)
    original_data = predict_data.copy()
    
    # Preprocesar datos (usando los mismos encoders que en entrenamiento)
    X, _ = preprocess_data(predict_data)
    for col in ['Sex', 'Embarked']:
        le = label_encoders[col]
        # Manejar categorías no vistas durante el entrenamiento
        X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    # Hacer predicciones
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Crear DataFrame con resultados
    results = original_data[['PassengerId', 'Name']].copy()
    results['Survived_Prediction'] = predictions
    results['Survival_Probability'] = probabilities
    
    return results

if __name__ == "__main__":
    # Archivos (cambia estas rutas según sea necesario)
    train_file = 'train.csv'  # Archivo con datos de entrenamiento (con columna Survived)
    predict_file = 'test.csv'  # Archivo con datos a predecir (sin columna Survived)
    
    # Entrenar modelo
    print("Entrenando modelo...")
    model, label_encoders = train_model(train_file)
    
    # Hacer predicciones
    print("\nRealizando predicciones...")
    predictions = predict_survival(predict_file, model, label_encoders)
    
    # Mostrar resultados
    print("\nResultados de predicción:")
    print(predictions.head())
    
    # Guardar resultados
    predictions.to_csv('titanic_predictions.csv', index=False)
    print("\nPredicciones guardadas en 'titanic_predictions.csv'")