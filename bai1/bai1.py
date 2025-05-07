import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Function to extract title from name
def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    rare_titles = ['Dr', 'Rev', 'Major', 'Col', 'Capt', 'Jonkheer', 'Don', 'Sir', 'Lady', 'Countess', 'Dona']
    return 'Rare' if title in rare_titles else title

# Function to preprocess data
def preprocess_data(df):
    # Create family_size and is_alone
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    
    # Extract title
    df['title'] = df['Name'].apply(extract_title)
    
    # Extract deck from cabin
    df['deck'] = df['Cabin'].str[0]
    df['deck'] = df['deck'].fillna('Unknown')
    
    # Calculate fare_per_person
    df['fare_per_person'] = df['Fare'] / df['family_size']
    
    return df

# Preprocess the data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Function to impute missing age using Linear Regression
def impute_age(df):
    # Prepare data for age prediction
    age_data = df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch']].copy()
    age_data['Sex'] = age_data['Sex'].map({'male': 0, 'female': 1})
    
    # Split into known and unknown age data
    known_age = age_data[age_data['Age'].notna()]
    unknown_age = age_data[age_data['Age'].isna()]
    
    # Train Linear Regression model
    X = known_age.drop('Age', axis=1)
    y = known_age['Age']
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Predict missing ages
    missing_age = lr.predict(unknown_age.drop('Age', axis=1))
    df.loc[df['Age'].isna(), 'Age'] = missing_age
    
    return df

# Function to impute missing embarked using KNN
def impute_embarked(df):
    # Prepare data for embarked prediction
    embarked_data = df[['Embarked', 'Pclass', 'Fare', 'Sex']].copy()
    embarked_data['Sex'] = embarked_data['Sex'].map({'male': 0, 'female': 1})
    
    # Split into known and unknown embarked data
    known_embarked = embarked_data[embarked_data['Embarked'].notna()]
    unknown_embarked = embarked_data[embarked_data['Embarked'].isna()]
    
    # Train KNN model
    X = known_embarked.drop('Embarked', axis=1)
    y = known_embarked['Embarked']
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    
    # Predict missing embarked
    missing_embarked = knn.predict(unknown_embarked.drop('Embarked', axis=1))
    df.loc[df['Embarked'].isna(), 'Embarked'] = missing_embarked
    
    return df

# Apply imputation
train_data = impute_age(train_data)
train_data = impute_embarked(train_data)

# Create boxplot for fare_per_person
plt.figure(figsize=(12, 6))
sns.boxplot(x='Pclass', y='fare_per_person', hue='Survived', data=train_data)
plt.title('Fare per Person Distribution by Pclass and Survival')
plt.savefig('fare_per_person_boxplot.png')
plt.close()

# Statistical test for fare_per_person
survived_fare = train_data[train_data['Survived'] == 1]['fare_per_person']
not_survived_fare = train_data[train_data['Survived'] == 0]['fare_per_person']
t_stat, p_value = stats.ttest_ind(survived_fare, not_survived_fare)
print(f"T-test results for fare_per_person: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

# Prepare data for modeling
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'deck', 'family_size', 'is_alone', 'title']
X = train_data[features]
y = train_data['Survived']

# Create preprocessing pipeline
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'family_size']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'deck', 'is_alone', 'title']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create and train Random Forest model with Grid Search
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Print best parameters
print("Best parameters:", grid_search.best_params_)

# Evaluate model using cross-validation
cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Get feature importance
feature_names = (numeric_features + 
                grid_search.best_estimator_.named_steps['preprocessor']
                .named_transformers_['cat']
                .get_feature_names_out(categorical_features).tolist())

importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_

# Create feature importance plot
plt.figure(figsize=(12, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Print additional metrics
y_pred = grid_search.best_estimator_.predict(X)
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall: {recall_score(y, y_pred):.4f}")
print(f"F1-score: {f1_score(y, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y, y_pred):.4f}")
