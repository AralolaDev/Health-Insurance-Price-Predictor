import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

medical = pd.read_csv('/Users/aralolabamise-alabi/Downloads/insurance.csv')

numeric = ['age', 'bmi', 'children']
text = ['region', 'sex', 'smoker']
target = ['charges']

X = medical[numeric + text]
y = medical['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y , test_size= 0.2, random_state = 42
)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), text),
        ('num', StandardScaler(), numeric),
    ]
)

model_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

print('Training model pipeline...')
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print('--- MODEL RESULTS ---')
print(f'R² Accuracy Score: {r2_score(y_test, y_pred) * 100:.1f}%')
print(f'Average Error Margin (MAE): ${mean_absolute_error(y_test, y_pred):,.2f}')

import joblib

joblib.dump(model_pipeline, 'insurance_model.pkl')
print('\nSuccess! Model saved as "insurance_model.pkl"')