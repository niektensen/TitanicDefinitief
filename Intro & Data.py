import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

#Intro tekst
st.set_page_config(
    page_title="Intro en Data", 
    layout="wide"
)

st.title("Introductie en Data")

with st.container(border=True):
    st.markdown("""
        ### Introductie
        Dit dashboard is ontwikkeld als verbeterde versie voor Case 1. Het doel is om met behulp van data-analyse en visualisatie inzicht te krijgen in Titanic data,
        en vervolgens een voorspellend model op te bouwen.

        **Projectgroep 6:**  
        - Joris Kroone  
        - Niek Tensen  
                
        De hoofdvraag van dit project is: **"Welke factoren bepaalden de overlevingskans van een passagier?"**

        ### Veranderingen
        In dit dashboard zijn ten opzichte van de vorige case diverse veranderingen doorgevoerd.  
        Deze zullen wij zo inzichtelijk mogelijk maken.
    """
    )

#Data inladen en weergeven
df = pd.read_csv(r"C:\Users\niekt\Downloads\Titanic case 2.0\data\train.csv")

st.header("Data exploratie")

col1, col2, col3 = st.columns(3)
col1.metric("Aantal passagiers", df.shape[0])
col2.metric("Aantal variabelen", df.shape[1])
col3.metric("Aantal waarnemingen", df.count().sum())  # totaal aantal niet-NA waarden

st.markdown("---")

st.subheader("📋 Dataset")
st.dataframe(df)

#Missende waarden
st.subheader("🚨 Ontbrekende waarden per kolom")
with st.container(border=True):
    st.markdown("""
        Hieronder is een visualisatie te zien van het percentage ontbrekende waarden per kolom in de dataset.
        Dit helpt bij het identificeren van kolommen di aandacht nodig hebben tijdens de data cleaning fase.
                
        De volgende stappen zijn ondernomen om met de ontbrekende waarden om te gaan:
        1. **Cabin kolom**: Deze kolom is verwijderd vanwege het hoge percentage ontbrekende waarden.
        2. **Embarked kolom**: Missende waarden zijn ingevuld met de modus (meest voorkomende waarde).
        3. **Age kolom**: Missende waarden zijn geschat op basis van de mediane leeftijd binnen groepen gedefinieerd door 'Sex', 'Pclass' en 'Embarked'.  
    """
    )

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

# Data inladen
df = pd.read_csv(r"C:\Users\niekt\Downloads\Titanic case 2.0\data\train.csv")

# 1️⃣ Cabin kolom droppen
df = df.drop(columns=['Cabin'])

# 2️⃣ Embarked missende waarden invullen met modus
embarked_mode = df['Embarked'].mode()[0]  # modus berekenen
df['Embarked'] = df['Embarked'].fillna(embarked_mode)

# 3️⃣ Age missende waarden schatten
# Maak een functie die Age invult op basis van median van groep (Sex, Pclass, Embarked)
def fill_age(row):
    if pd.isna(row['Age']):
        median_age = df[
            (df['Sex'] == row['Sex']) &
            (df['Pclass'] == row['Pclass']) &
            (df['Embarked'] == row['Embarked'])
        ]['Age'].median()
        return median_age
    else:
        return row['Age']

df['Age'] = df.apply(fill_age, axis=1)

# Check dat alles is opgevuld
print("Aantal missende waarden per kolom:")
print(df.isna().sum())

# Voeg kolom 'Alleenreiziger' toe
df["Alleenreiziger"] = df.apply(
lambda x: "Ja" if x["SibSp"] == 0 and x["Parch"] == 0 else "Nee", axis=1
)

# Voeg kolom 'Titel' toe (haal het stukje tussen komma en punt)
df["Titel"] = df["Name"].str.extract(r",\s*([A-Za-z]+)\.")

# Toon de relevante kolommen
st.subheader("✨ Nieuw toegevoegde data")
with st.container(border=True):
    st.markdown("""
        De volgende kolommen zijn toegevoegd aan de dataset na het opschonen:
        - **Titel**: Afgeleid uit de naam van de passagier, geeft een indicatie van sociale status en geslacht.
        - **Alleenreiziger**: Geeft aan of de passagier alleen reisde (geen familie aan boord).
    """
    )
st.dataframe(df[["Name", "Titel", "Alleenreiziger"]])

    





















