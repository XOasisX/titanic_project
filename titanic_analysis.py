import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('C:/Users/Estudiante_MCA/Desktop/taller_jenkins/data/train.csv')
df = df.dropna(subset=['Age', 'Sex', 'Survived'])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
X = df[['Age', 'Sex']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
df['prediction'] = model.predict(X)
df[['PassengerId', 'prediction']].to_csv('resultados.csv', index=False)