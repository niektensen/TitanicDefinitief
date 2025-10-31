import streamlit as st
import pandas as pd
import numpy as np
import load_data
import re
from io import StringIO
import plotly.graph_objects as go
 
# Machine Learning libraries
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
 
st.set_page_config(page_title="Modelleren", layout="wide", page_icon="🤖")
st.title("🤖 Model Vergelijking & Voorspelling")
 
# Laad zowel de ruwe als de schone data
train_raw, test_raw = load_data.load_raw_data()
train_cleaned, test_cleaned = load_data.get_cleaned_data()
 
# Check of alle data beschikbaar is
if train_raw is None or train_cleaned is None:
    st.warning("Data kon niet volledig geladen worden. Controleer of `run_preprocessing.py` is uitgevoerd.")
    st.stop()
 
# Maak een 5-fold cross-validation voor model evaluatie
kf = KFold(n_splits=5, shuffle=True, random_state=42)
st.markdown("---")
 
# --- Maak de 2 kolommen layout ---
col1, col2 = st.columns(2)
 
# --- Kolom 1: Oude Versie (Hard-coded) ---
with col1:
    st.header("Oude Versie")
 
    @st.cache_data
    def run_old_model(train_df_raw):
        # Eenvoudige baseline versie met beperkte features
        train_df = train_df_raw.copy()
       
        try:
            test_df = pd.read_csv("data/test.csv")
        except FileNotFoundError:
            return None, "Test.csv niet gevonden"
       
        combined_data = pd.concat([train_df.drop('Survived', axis=1), test_df], axis=0)
 
        # Gebruik alleen Pclass, Sex en Fare als features
        combined_data['Sex'] = combined_data['Sex'].map({'female': 0, 'male': 1})
        combined_data['Fare'] = combined_data['Fare'].fillna(combined_data['Fare'].median())
 
        # Verwijder overige kolommen
        combined_data = combined_data.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Embarked', 'Title', 'Age'], axis=1, errors='ignore')
 
        # Splits terug naar train en test
        proc_train = combined_data[:len(train_df)]
        proc_test = combined_data[len(train_df):]
 
        # Maak features klaar voor model training
        trn_X = proc_train.drop('PassengerId', axis=1)
        trn_Y = train_df['Survived']
       
        # Initialiseer en train het Random Forest model
        forest_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
 
        # Cross-validatie om de prestaties te beoordelen
        kf_old = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_accuracy_scores = cross_val_score(forest_model, trn_X, trn_Y, cv=kf_old, scoring='accuracy')
       
        return cv_accuracy_scores.mean(), None
 
    # Voer de oude methode uit
    old_score, error = run_old_model(train_raw)
   
    if error:
        st.error(error)
    else:
        st.metric(
            label="Baseline Accuracy",
            value=f"{old_score:.4f}"
        )
 
 
# --- Kolom 2: Nieuwe Versie (Interactieve App) ---
with col2:
    st.header("Nieuwe Versie")
 
    # Definieer de features die we gaan gebruiken
    numeric_features = ['Age', 'Fare', 'FamilySize', 'IsAlone']
    categorical_features = ['Embarked', 'Sex', 'Pclass', 'Title']
 
    # --- Functie om de 'Nieuwe' Pipeline te bouwen ---
    @st.cache_resource
    def build_new_pipeline(_model): # Argument '_model' om caching error te voorkomen
       
        # 1. Numerieke pipeline: Standaardiseer de data
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
 
        # 2. Categorische pipeline: One-Hot Encode de data
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
 
        # 3. Combineer de pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
 
        # 4. Maak de volledige pipeline
        clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', _model)]) # Gebruik '_model'
        return clf_pipeline
 
    # --- Bereid de data voor ---
    @st.cache_data
    def prepare_training_data():
        X_train = train_cleaned.copy()
        X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch'] + 1
        X_train['IsAlone'] = (X_train['FamilySize'] == 1).astype(int)
        y_train = X_train['Survived']
        return X_train, y_train
   
    X_train, y_train = prepare_training_data()
 
    # --- Definieer onze beste parameters ---
    best_rf_params = {
        'n_estimators': 47,
        'max_depth': 10,
        'min_samples_leaf': 4,
        'random_state': 1
    }
   
    # --- Functie: Train en cache het beste model ---
    @st.cache_resource
    def get_trained_model():
        best_rf_model = RandomForestClassifier(**best_rf_params)
        final_model = build_new_pipeline(best_rf_model)
        final_model.fit(X_train, y_train)
        return final_model
   
    # --- Functie: Haal mediaan Fares op ---
    @st.cache_data
    def get_median_fares():
        return train_cleaned.groupby('Pclass')['Fare'].median()
 
    # --- Model Vergelijking (Teruggezet) ---
    st.subheader("Model Vergelijking")
   
    # --- FIX: Verwijder keuzemenu en knop, toon RF-score direct ---
    with st.spinner("Validating Random Forest"):
        rf_pipeline = build_new_pipeline(RandomForestClassifier(**best_rf_params))
        scores_rf = cross_val_score(rf_pipeline, X_train, y_train, cv=kf, scoring='accuracy')
        st.metric(
            label=f"Nieuw: Random Forest",
            value=f"{scores_rf.mean():.4f}",
            delta=f"{(scores_rf.mean() - (old_score or 0)):.4f} vs. Oude Baseline"
        )
    # --- EINDE FIX ---
   
    st.markdown("---") # Scheidingslijn
   
    # --- Finale Voorspelling (Download knop) ---
    st.subheader("Voorspelling")
   
    if st.button("Genereer `submission.csv`"):
        with st.spinner("Model trainen op alle data en voorspellen..."):
           
            # Bereid de testdata voor (met 'IsAlone')
            X_test = test_cleaned.copy()
            X_test['FamilySize'] = X_test['SibSp'] + X_test['Parch'] + 1
            X_test['IsAlone'] = (X_test['FamilySize'] == 1).astype(int)
           
            # st.success("Beste model wordt gebruikt!") # Verwijderd
            final_model = get_trained_model() # Haalt het gecachte, getrainde model op
           
            # Voorspel op test data
            predictions = final_model.predict(X_test)
           
            # Maak het submission DataFrame
            submission_df = pd.DataFrame({
                "PassengerId": test_cleaned["PassengerId"],
                "Survived": predictions.astype(int)
            })
           
            # Maak het CSV-bestand downloadbaar
            csv = submission_df.to_csv(index=False).encode('utf-8')
           
            st.download_button(
                label="Download submission.csv",
                data=csv,
                file_name="titanic_submission.csv",
                mime="text/csv",
            )
            st.success("`submission.csv` is klaar om te downloaden!")
 
# Functie voor het maken van vloeiende kleurovergangen
def get_smooth_color(value_0_to_1):
    # Definieer de basiskleuren
    Rood = (214, 39, 40)    # Laag risico
    Geel = (240, 229, 29)   # Gemiddeld risico
    Groen = (44, 160, 44)   # Hoog risico
   
    # Bereken de kleurovergang
    if value_0_to_1 <= 0.5:
        # Van rood naar geel
        t = value_0_to_1 * 2
        r = int((1 - t) * Rood[0] + t * Geel[0])
        g = int((1 - t) * Rood[1] + t * Geel[1])
        b = int((1 - t) * Rood[2] + t * Geel[2])
    else:
        # Van geel naar groen
        t = (value_0_to_1 - 0.5) * 2
        r = int((1 - t) * Geel[0] + t * Groen[0])
        g = int((1 - t) * Geel[1] + t * Groen[1])
        b = int((1 - t) * Geel[2] + t * Groen[2])
   
    return f'rgb({r}, {g}, {b})'
 
# Interactieve tool voor het voorspellen van overlevingskansen
st.markdown("---")
st.header("🤖 Interactieve Voorspeller")
 
# Maak twee kolommen voor input en resultaat
pred_col1, pred_col2 = st.columns([1, 1.5])
 
with pred_col1:
    st.markdown("##### Passagiersgegevens")
    with st.container(border=True):
        # Haal mediaan ticketprijzen op voor elke klasse
        median_fares = get_median_fares()
       
        # Invoervelden voor passagierskenmerken
        p_geslacht_str = st.selectbox("Geslacht:", ('Man', 'Vrouw'), key='pred_sex')
        p_klasse_str = st.selectbox("Klasse:", ('1e Klasse', '2e Klasse', '3e Klasse'), key='pred_class')
        p_haven_str = st.selectbox("Haven:", ('Southampton', 'Cherbourg', 'Queenstown'), key='pred_embark')
        p_leeftijd = st.slider("Leeftijd:", 0, 80, 25, key='pred_age')
        p_familie = st.slider("Totaal Aantal Familieleden (incl. uzelf):", 1, 11, 1, key='pred_family')
 
# Verwerk de ingevoerde gegevens
model = get_trained_model()
 
# Bereken afgeleide kenmerken
p_is_alone = 1 if p_familie == 1 else 0
p_pclass = int(p_klasse_str[0])
p_sex = 'male' if p_geslacht_str == 'Man' else 'female'
p_embarked = p_haven_str[0]
 
# Bepaal de juiste titel op basis van geslacht en leeftijd
if p_sex == 'male':
    p_title = 'Master' if p_leeftijd <= 12 else 'Mr'
else:
    p_title = 'Miss' if p_leeftijd <= 18 else 'Mrs'
   
# Gebruik de mediaan ticketprijs voor de gekozen klasse
p_fare = median_fares.get(p_pclass, train_cleaned['Fare'].median())
 
input_data = pd.DataFrame({
    'Age': [p_leeftijd],
    'Fare': [p_fare],
    'FamilySize': [p_familie],
    'IsAlone': [p_is_alone],
    'Embarked': [p_embarked],
    'Sex': [p_sex],
    'Pclass': [p_pclass],
    'Title': [p_title],
    'SibSp': [0],
    'Parch': [0]
})
 
prediction_proba = model.predict_proba(input_data)
kans_overleefd_float = prediction_proba[0][1]
kans_overleefd_pct = kans_overleefd_float * 100
 
bar_color = get_smooth_color(kans_overleefd_float)
# Toon de voorspelling
with pred_col2:
    st.markdown("##### Voorspelling Resultaat")
    with st.container(border=True):
       
        # Maak een meter-visualisatie voor de overlevingskans
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = kans_overleefd_pct,
            number = {'suffix': "%", 'font': {'size': 40}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Voorspelde Overlevingskans", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': bar_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps' : [
                    {'range': [0, 50], 'color': 'white'},
                    {'range': [50, 100], 'color': '#EFEFEF'}
                ],
                'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
       
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
       
        st.plotly_chart(fig, use_container_width=True)
       
        if kans_overleefd_float > 0.5:
            st.success("Deze passagier zou de ramp **waarschijnlijk overleven**.")
        else:
            st.error("Deze passagier zou **waarschijnlijk overlijden**.")
 
with st.expander("Bekijk de data die naar het model is gestuurd"):
    st.dataframe(input_data)
 
