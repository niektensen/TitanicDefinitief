import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

# --- Definieer de bestandspaden ---
DATA_DIR = "data"
TRAIN_INPUT = os.path.join(DATA_DIR, "train.csv")
TEST_INPUT = os.path.join(DATA_DIR, "test.csv")

TRAIN_OUTPUT = os.path.join(DATA_DIR, "train_processed.csv")
TEST_OUTPUT = os.path.join(DATA_DIR, "test_processed.csv")

def preprocess_titanic_data(train_path=TRAIN_INPUT, test_path=TEST_INPUT):
    """
    Laadt, verwerkt en slaat de Titanic datasets op basis van de
    gespecificeerde vereisten.
    """
    
    # --- Stap 1: Laad de datasets ---
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        print(f"Bestanden '{train_path}' en '{test_path}' succesvol geladen.")
    except FileNotFoundError as e:
        print(f"Fout: {e}. Zorg ervoor dat de bestanden in de map '{DATA_DIR}' staan.")
        return

    # Bewaar de IDs en 'Survived' kolom voor later
    train_ids = df_train['PassengerId']
    test_ids = df_test['PassengerId']
    # Kopieer Survived kolom om SettingWithCopyWarning te vermijden
    survived_col = df_train['Survived'].copy()

    # --- Stap 2: 'Cabin' Verwijderen ---
    df_train = df_train.drop('Cabin', axis=1)
    df_test = df_test.drop('Cabin', axis=1)
    print("Kolom 'Cabin' verwijderd uit beide datasets.")

    # --- Stap 3: 'Fare' en 'Embarked' Imputeren ---
    
    # Imputeer 'Fare' (mediaan van de Pclass)
    # Iets robuuster: mediaan per Pclass
    df_train['Fare'] = df_train.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    df_test['Fare'] = df_test.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    # Noodoplossing als een Pclass nog steeds NaN heeft
    df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].median())
    df_test['Fare'] = df_test['Fare'].fillna(df_train['Fare'].median())
    print("Ontbrekende 'Fare'-waarden aangevuld met mediaan per Pclass.")

    # Imputeer 'Embarked' (modus)
    embarked_mode = df_train['Embarked'].mode()[0]
    df_train['Embarked'] = df_train['Embarked'].fillna(embarked_mode)
    df_test['Embarked'] = df_test['Embarked'].fillna(embarked_mode)
    print(f"Ontbrekende 'Embarked'-waarden aangevuld met modus: {embarked_mode}")

    # --- Stap 4: 'Age' Voorspellingsmodel ---

    # Combineer de datasets (zonder 'Survived') voor consistente feature engineering
    df_train_temp = df_train.drop('Survived', axis=1)
    df_combined = pd.concat([df_train_temp, df_test], ignore_index=True)

    # 4a. Feature Engineering: Haal 'Title' uit 'Name'
    df_combined['Title'] = df_combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Vervang zeldzame titels
    df_combined['Title'] = df_combined['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_combined['Title'] = df_combined['Title'].replace('Mlle', 'Miss')
    df_combined['Title'] = df_combined['Title'].replace('Ms', 'Miss')
    df_combined['Title'] = df_combined['Title'].replace('Mme', 'Mrs')
    print("Feature 'Title' gecreëerd en zeldzame titels gegroepeerd.")

    # 4b. Prepareer data voor het model
    # We droppen kolommen die we niet gebruiken voor de voorspelling van leeftijd
    features_for_age_model = df_combined.drop(['Name', 'Ticket', 'PassengerId'], axis=1)

    # Converteer categorische variabelen naar dummy-variabelen
    features_encoded = pd.get_dummies(features_for_age_model, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

    # 4c. Splits data in sets voor trainen en voorspellen
    age_known = features_encoded[features_encoded['Age'].notnull()]
    age_unknown = features_encoded[features_encoded['Age'].isnull()]

    # Definieer X en y voor het leeftijdsvoorspellingsmodel
    X_age_train = age_known.drop('Age', axis=1)
    y_age_train = age_known['Age']
    X_age_predict = age_unknown.drop('Age', axis=1)

    print(f"Data voorbereid. We trainen het 'Age'-model op {len(X_age_train)} rijen.")

    # 4d. Train het model
    rf_age = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_age.fit(X_age_train, y_age_train)
    print("Random Forest Regressor getraind om 'Age' te voorspellen.")

    # 4e. Voorspel en vul de ontbrekende leeftijden in
    predicted_ages = rf_age.predict(X_age_predict)

    # Vul de waarden in de *gecombineerde* dataset in
    df_combined.loc[df_combined['Age'].isnull(), 'Age'] = predicted_ages
    print(f"Ontbrekende 'Age'-waarden ({len(predicted_ages)} stuks) voorspeld en ingevuld.")

    # --- Stap 5: Opslaan van de opgeschoonde data ---

    # Drop de helper features ('Name', 'Ticket')
    # We BEHOUDEN 'Title' omdat dat een nuttige feature is voor het hoofdmodel
    df_combined_final = df_combined.drop(['Name', 'Ticket'], axis=1)

    # Splits de data terug in train en test
    df_train_processed = df_combined_final[df_combined_final['PassengerId'].isin(train_ids)]
    df_test_processed = df_combined_final[df_combined_final['PassengerId'].isin(test_ids)]

    # Voeg 'Survived' terug toe aan de trainingsset
    # Gebruik .loc om SettingWithCopyWarning te vermijden
    df_train_processed.loc[:, 'Survived'] = survived_col.values
    
    # Sla de bestanden op
    df_train_processed.to_csv(TRAIN_OUTPUT, index=False)
    df_test_processed.to_csv(TEST_OUTPUT, index=False)

    print(f"\nVerwerking voltooid.")
    print(f"Verwerkte bestanden opgeslagen als '{TRAIN_OUTPUT}' en '{TEST_OUTPUT}'.")
    
    print("\n--- Info voor train_processed.csv ---")
    df_train_processed.info()
    
    print("\n--- Info voor test_processed.csv ---")
    df_test_processed.info()


# --- Voer de functie uit ---
if __name__ == "__main__":
    preprocess_titanic_data()
