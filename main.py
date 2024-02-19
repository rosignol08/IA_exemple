from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

fnlwgt_ = []
file_path = 'donee/adult.test'
file_path_data = 'donee/adult.data'
with open(file_path, 'r') as file:
    for line in file:
        split_data = line.strip().split(', ')
        X = 2
        if len(split_data) > X:
            data_after_Xth_comma = split_data[X]
            fnlwgt_.append(data_after_Xth_comma)

#print(fnlwgt_)


#entrainement 'test'
data = []
with open(file_path, 'r') as file:
    if '|' in line:
        next(file)  # Ignorer la première ligne (entête)
    else:
        for line in file:  
            line = line.strip().split(', ')
            if '?' not in line:
                if len(line) >= 11:
                    data.append({'age': int(line[0]), 'workclass': str(line[1]), 'fnlwgt': int(line[2]),'education':str(line[3]), 'education-num': int(line[4]), 'marital-status': str(line[5]), 'occupation': str(line[6]), 'relationship': str(line[7]), 'race': str(line[8]), 'sex': str(line[9]), 'capital-gain': int(line[10]), 'capital-loss': int(line[11]), 'hours-per-week': int(line[12]), 'native-country': str(line[13]), 'income': str(line[14])})

df = pd.DataFrame(data)

df = pd.get_dummies(df, columns=['age'])
df = pd.get_dummies(df, columns=['workclass'])
df = pd.get_dummies(df, columns=['fnlwgt'])
df = pd.get_dummies(df, columns=['education'])
df = pd.get_dummies(df, columns=['education-num'])
df = pd.get_dummies(df, columns=['marital-status'])
df = pd.get_dummies(df, columns=['occupation'])
df = pd.get_dummies(df, columns=['relationship'])
df = pd.get_dummies(df, columns=['race'])
df = pd.get_dummies(df, columns=['sex'])
df = pd.get_dummies(df, columns=['capital-gain'])
df = pd.get_dummies(df, columns=['capital-loss'])
df = pd.get_dummies(df, columns=['hours-per-week'])
df = pd.get_dummies(df, columns=['native-country'])
df = pd.get_dummies(df, columns=['income'])

print(df.columns)

# Diviser les données en caractéristiques (X) et étiquettes (y)
X = df.drop(['income_<=50K.', 'income_>50K.'], axis=1)
y = df['income_>50K.']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

# Évaluer la précision du modèle
accuracy = knn.score(X_test, y_test)
print("Précision du modèle de test: {:.2f}%".format(accuracy * 100))



# Charger les données de "adult.data"
data_final = []
with open(file_path_data, 'r') as file_data:
    for line in file_data:  
        line = line.strip().split(', ')
        if '?' not in line:
            if len(line) >= 11:
                data_final.append({'age': int(line[0]), 'workclass': str(line[1]), 'fnlwgt': int(line[2]),'education':str(line[3]), 'education-num': int(line[4]), 'marital-status': str(line[5]), 'occupation': str(line[6]), 'relationship': str(line[7]), 'race': str(line[8]), 'sex': str(line[9]), 'capital-gain': int(line[10]), 'capital-loss': int(line[11]), 'hours-per-week': int(line[12]), 'native-country': str(line[13]), 'income': str(line[14])})

# Créer un DataFrame pour les données de "adult.data"
df_data = pd.DataFrame(data_final)

df_data = pd.get_dummies(df_data, columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
#df_data.drop(['income_<=50K', 'income_>50K'], axis=1, inplace=True)
#print(df_data.columns)
X_ready = df_data.drop(['income_<=50K', 'income_>50K'], axis=1)
y_ready = df_data['income_>50K']
training_columns = X_train.columns

training_columns = X.columns.tolist()  # X est la variable des caractéristiques utilisées pour l'entraînement

# Assure-toi que df_test contient les mêmes colonnes que celles utilisées lors de l'entraînement
df_data_adjusted = df[training_columns]

columns_to_drop = [col for col in df_data_adjusted.columns if col not in training_columns]
df_data_adjusted = df_data_adjusted.drop(columns=columns_to_drop, axis=1)

# Sélection des mêmes colonnes dans l'ensemble de test
#X_ready_adjusted = X_ready[training_columns]

# Utiliser le modèle entraîné sur "adult.test" pour faire des prédictions sur les données de "adult.data"
predictions_data = knn.predict(df_data_adjusted)
# Utiliser le modèle entraîné sur "adult.test" pour faire des prédictions sur les données de "adult.data"
#predictions_data = knn.predict(X_ready)
predictions_list = predictions_data.tolist()
print("Précision du modèle final: {:.2f}%".format(predictions_list[0] * 100))