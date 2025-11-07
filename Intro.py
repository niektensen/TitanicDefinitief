import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Pagina-instellingen
st.set_page_config(
    page_title="Introductie en Data",
    layout="wide"
)

st.title("Introductie en Data")

# Inleidende tekst
with st.container(border=True):
    st.markdown("""
    ### Introductie
    Dit dashboard is ontwikkeld als verbeterde versie van Case 1. 
    Het doel is om met behulp van data-analyse en visualisatie inzicht te krijgen in de Titanic-data,
    en vervolgens een voorspellend model op te bouwen.

    **Projectgroep 6:**  
    - Joris Kroone  
    - Niek Tensen  

    De hoofdvraag van dit project is:  
    **Welke factoren bepaalden de overlevingskans van een passagier?**

    ### Veranderingen
    In dit dashboard zijn ten opzichte van de vorige case diverse verbeteringen doorgevoerd.
    Deze worden hieronder toegelicht.
    """)

# Data inladen
df = pd.read_csv("data/train.csv")

# Basisinformatie
st.header("Data-exploratie")

col1, col2, col3 = st.columns(3)
col1.metric("Aantal passagiers", df.shape[0])
col2.metric("Aantal variabelen", df.shape[1])
col3.metric("Aantal waarnemingen", df.count().sum())

st.markdown("---")

# Dataset weergeven
st.subheader("Dataset")
st.dataframe(df)

# Ontbrekende waarden
st.subheader("Ontbrekende waarden per kolom")

with st.container(border=True):
    st.markdown("""
    Onderstaande grafiek toont het percentage ontbrekende waarden per kolom. 
    Dit geeft inzicht in kolommen die aandacht vragen tijdens de opschoningsfase.

    **Aanpak ontbrekende waarden:**
    1. *Cabin* is verwijderd vanwege het hoge percentage ontbrekende waarden.  
    2. *Embarked* is ingevuld met de modus (meest voorkomende waarde).  
    3. *Age* is geschat op basis van de mediane leeftijd binnen groepen gedefinieerd door 'Sex', 'Pclass' en 'Embarked'.
    """)

missing = df.isnull().sum().reset_index()
missing.columns = ["Kolom", "Aantal_ontbrekend"]
missing["Percentage"] = (missing["Aantal_ontbrekend"] / len(df) * 100).round(1)

fig_missing = px.bar(
    missing,
    x="Kolom",
    y="Percentage",
    color="Percentage",
    title="Percentage ontbrekende waarden per kolom",
    color_continuous_scale="Reds"
)
st.plotly_chart(fig_missing, use_container_width=True)

# Data opschonen
st.header("Data opschonen")

# Cabin verwijderen
df = df.drop(columns=["Cabin"])

# Embarked invullen met modus
embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)

# Age invullen met mediane leeftijd per groep
def fill_age(row):
    if pd.isna(row["Age"]):
        median_age = df[
            (df["Sex"] == row["Sex"]) &
            (df["Pclass"] == row["Pclass"]) &
            (df["Embarked"] == row["Embarked"])
        ]["Age"].median()
        return median_age
    return row["Age"]

df["Age"] = df.apply(fill_age, axis=1)

# Controle op missende waarden
st.write("Aantal missende waarden per kolom na opschoning:")
st.dataframe(df.isna().sum().to_frame("Aantal"))

# Nieuwe kolommen toevoegen
st.header("Toegevoegde kolommen")

# Alleenreiziger
df["Alleenreiziger"] = np.where((df["SibSp"] == 0) & (df["Parch"] == 0), "Ja", "Nee")

# Titel uit naam
df["Titel"] = df["Name"].str.extract(r",\s*([A-Za-z]+)\.")

with st.container(border=True):
    st.markdown("""
    **Nieuw toegevoegde kolommen:**
    - *Titel*: Afgeleid uit de naam van de passagier, geeft een indicatie van sociale status en geslacht.  
    - *Alleenreiziger*: Geeft aan of de passagier alleen reisde (geen familie aan boord).
    """)

st.dataframe(df[["Name", "Titel", "Alleenreiziger"]])

    





















