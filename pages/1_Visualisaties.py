import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from statsmodels.stats.proportion import proportion_confint
import sys
 
sys.path.append(("data/train.csv"))  # Zorg dat load_data.py vindbaar is (indien nodig)
 
# Data inladen
df = pd.read_csv(("data/train.csv"))
 
# 1 - Cabin kolom droppen
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])
 
# 2 - Embarked missende waarden invullen met modus
if 'Embarked' in df.columns:
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
 
# 3 - Age missende waarden schatten op basis van median per groep (Sex, Pclass, Embarked)
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
 
if 'Age' in df.columns:
    df['Age'] = df.apply(fill_age, axis=1)
 
# Check voor missende waarden (optioneel debugging)
# print("Aantal missende waarden per kolom:")
# print(df.isna().sum())
 
# Voeg kolom 'Alleenreiziger' / 'IsAlone' toe
df["Alleenreiziger"] = df.apply(lambda x: "Ja" if (x.get("SibSp", 0) == 0 and x.get("Parch", 0) == 0) else "Nee", axis=1)
df["IsAlone"] = ((df.get("SibSp", 0) + df.get("Parch", 0)) == 0).astype(int)
 
# Voeg kolom 'Titel' / 'Title' toe (haal het stukje tussen komma en punt)
df["Titel"] = df["Name"].str.extract(r",\s*([A-Za-z]+)\.")[0].str.strip()
# Maak ook een Engelse 'Title' kolom voor compatibiliteit met verschillende secties
df["Title"] = df["Titel"]
 
# Standaard pagina-instellingen
st.set_page_config(page_title="Titanic Dashboard Visualisaties", layout="wide")
 
# === Sidebar: keuze oud of nieuwe visualisaties ===
visual_choice = st.sidebar.radio(
    "Kies visualisatieversie:",
    ("Nieuwe visualisaties", "Oude visualisaties"),
    index=0  # Standaard: Nieuwe visualisaties
)
 
# ===================== NIEUWE VISUALISATIES =====================
if visual_choice == "Nieuwe visualisaties":
    st.header("Nieuwe Visualisaties")
 
    # Extra kolommen (Nederlandse labels)
    df["Geslacht"] = df["Sex"].map({'male': "Man", 'female': "Vrouw"})
    df["Overleefd"] = df["Survived"].map({0: "Nee", 1: "Ja"})
    df["Klasse"] = df["Pclass"].map({1: "Klasse 1", 2: "Klasse 2", 3: "Klasse 3"})
    df["Opstaphaven"] = df["Embarked"].map({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"})
    # 'Alleenreiziger' en 'Titel' zijn reeds aangemaakt hierboven
 
    # Container voor hoofdgedeelte (houdt layout netjes)
    with st.container():
        feature = st.selectbox(
            "Kies een variabele:",
            ["Geslacht", "Klasse", "Opstaphaven", "Titel", "Alleenreiziger"]
        )
 
        y_as = st.radio("Y-as weergeven als:", ["Percentage", "Aantal personen"], horizontal=True)
        show_error = st.checkbox("Toon 95% betrouwbaarheidsinterval (alleen bij %)", value=False)
 
        # Data aggregeren
        agg_data = df.groupby([feature, "Overleefd"]).size().reset_index(name="count")
        total_per_cat = df.groupby([feature]).size().reset_index(name="total")
        agg_data = agg_data.merge(total_per_cat, on=feature)
        agg_data["perc"] = agg_data["count"] / agg_data["total"] * 100
 
        if show_error and y_as == "Percentage":
            lower, upper = [], []
            for _, row in agg_data.iterrows():
                ci_low, ci_upp = proportion_confint(count=row["count"], nobs=row["total"], alpha=0.05, method='wilson')
                lower.append(ci_low * 100)
                upper.append(ci_upp * 100)
            agg_data["error_y"] = [u - l for l, u in zip(lower, upper)]
        else:
            agg_data["error_y"] = None
 
        y_col = "perc" if y_as == "Percentage" else "count"
        y_label = "Percentage (%)" if y_as == "Percentage" else "Aantal personen"
 
        fig = px.bar(
            agg_data,
            x=feature,
            y=y_col,
            color="Overleefd",
            error_y="error_y" if y_as == "Percentage" else None,
            barmode="group",
            color_discrete_map={"Ja": "#00CC96", "Nee": "#EF553B"},
            labels={feature: feature, y_col: y_label, "Overleefd": "Overleefd"}
        )
 
        fig.update_traces(texttemplate='', textposition='none')
        fig.update_layout(
            height=500,
            xaxis_title=feature,
            yaxis_title=y_label,
            legend_title_text="Overleefd",
            legend=dict(bgcolor="white", bordercolor="LightGrey", borderwidth=1, font=dict(size=12)),
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
 
        st.plotly_chart(fig, use_container_width=True)
 
    # ======================
    # Scatterplot: Leeftijd vs Fare
    # ======================
    st.subheader("🎯 Leeftijd vs Ticketprijs – Interactieve Scatterplot")
 
    # Checkboxen bovenaan
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_male = st.checkbox("👨 Mannen", value=True)
    with col2:
        show_female = st.checkbox("👩 Vrouwen", value=True)
    with col3:
        show_survived = st.checkbox("✅ Overleefd", value=True)
    with col4:
        show_not_survived = st.checkbox("❌ Niet overleefd", value=True)
 
    # Filter toepassen
    filtered_df = df.copy()
    if not show_male:
        filtered_df = filtered_df[filtered_df["Geslacht"] != "Man"]
    if not show_female:
        filtered_df = filtered_df[filtered_df["Geslacht"] != "Vrouw"]
    if not show_survived:
        filtered_df = filtered_df[filtered_df["Overleefd"] != "Ja"]
    if not show_not_survived:
        filtered_df = filtered_df[filtered_df["Overleefd"] != "Nee"]
 
    # Scatterplot
    fig = px.scatter(
        filtered_df,
        x="Age",
        y="Fare",
        color="Overleefd",
        symbol="Geslacht",
        labels={
            "Age": "Leeftijd (jaren)",
            "Fare": "Ticketprijs (£)",
            "Overleefd": "Overleefd",
            "Geslacht": "Geslacht"
        },
        color_discrete_map={"Ja": "#00CC96", "Nee": "#EF553B"},
        symbol_map={"Man": "circle", "Vrouw": "diamond"},
        hover_data=["Name", "Pclass", "Embarked"]
    )
 
    fig.update_layout(
        height=600,
        xaxis_title="Leeftijd (jaren)",
        yaxis_title="Ticketprijs (£)",
        legend_title_text="Legenda",
        legend=dict(
            title="Uitleg",
            itemsizing='trace',
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='LightGrey',
            borderwidth=1
        )
    )
 
    st.plotly_chart(fig, use_container_width=True)
 
    # ======================
    # Hier voegen we jouw interactieve correlatie-matrix code toe
    # ======================
    st.markdown("---")
    st.header("Interactieve Correlatie Matrix")
    sns.set_style('whitegrid')
 
    # Werk met een kopie zodat we originele df niet per ongeluk aanpassen
    df_corr = df.copy()
 
    # Zorg dat Age en Fare geen missende waarden hebben voor de binning
    if 'Age' in df_corr.columns:
        df_corr['Age'] = df_corr['Age'].fillna(df_corr['Age'].median())
    if 'Fare' in df_corr.columns:
        df_corr['Fare'] = df_corr['Fare'].fillna(df_corr['Fare'].median())
 
    # Maak Age_bin en Fare_bin indien nog niet aanwezig
    if 'Age_bin' not in df_corr.columns:
        df_corr['Age_bin'] = pd.cut(df_corr['Age'], bins=[0, 12, 20, 40, 120], labels=['Kind', 'Tiener', 'Volwassen', 'Oud'])
    if 'Fare_bin' not in df_corr.columns:
        try:
            df_corr['Fare_bin'] = pd.qcut(df_corr['Fare'], 4, labels=['Laag', 'Gemiddeld', 'Hoog', 'Zeer hoog'])
        except ValueError:
            # fallback - gelijk verdeelde bins als qcut faalt
            df_corr['Fare_bin'] = pd.cut(df_corr['Fare'], bins=4, labels=['Laag', 'Gemiddeld', 'Hoog', 'Zeer hoog'])
 
    # FamilySize en IsAlone kolommen
    if 'FamilySize' not in df_corr.columns:
        df_corr['FamilySize'] = df_corr.get('SibSp', 0) + df_corr.get('Parch', 0) + 1
    if 'IsAlone' not in df_corr.columns:
        df_corr['IsAlone'] = (df_corr['FamilySize'] == 1).astype(int)
 
    # Maak dummy variabelen - let op: gebruik 'Title' (Engelse naam) als kolom voor compatibiliteit
    # Zorg dat 'Title' aanwezig is
    if 'Title' not in df_corr.columns:
        df_corr['Title'] = df_corr['Name'].str.extract(r",\s*([A-Za-z]+)\.")[0].str.strip()
 
    df_corr_dummied = pd.get_dummies(
        df_corr,
        columns=["Sex", "Age_bin", "Embarked", "Fare_bin", "Title", "Pclass"],
        prefix=["Sex", "Age_type", "Em_type", "Fare_type", "Title", "Pclass"],
        drop_first=False
    )
 
    # 2. Definieer de selectie-opties
    numeric_options = ['Survived', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'SibSp', 'Parch']
    dummy_options = ['Pclass', 'Sex', 'Embarked', 'Title', 'Age_bin', 'Fare_bin']
 
    # Mapping van de "basis" naam naar de dummy prefix
    dummy_prefixes = {
        'Pclass': 'Pclass_',
        'Sex': 'Sex_',
        'Embarked': 'Em_type_',
        'Title': 'Title_',
        'Age_bin': 'Age_type_',
        'Fare_bin': 'Fare_type_'
    }
 
    all_options = sorted(numeric_options + dummy_options)
 
    # 3. Maak het "slimme" multiselect menu
    selected_options = st.multiselect(
        "Kies variabelen voor de correlatiematrix:",
        options=all_options,
        default=['Survived', 'FamilySize', 'IsAlone', 'Title', 'Pclass']
    )
 
    # 4. Vertaal de selecties naar de uiteindelijke kolomlijst
    final_selected_cols = []
    for option in selected_options:
        if option in numeric_options:
            final_selected_cols.append(option)
        elif option in dummy_prefixes:
            prefix = dummy_prefixes[option]
            dummy_cols = [col for col in df_corr_dummied.columns if col.startswith(prefix)]
            final_selected_cols.extend(dummy_cols)
 
    # 5. Plot de heatmap
    if len(final_selected_cols) > 1:
        final_selected_cols = list(dict.fromkeys(final_selected_cols))
        existing_cols = [col for col in final_selected_cols if col in df_corr_dummied.columns]
 
        # Zorg dat 'Survived' numeriek is (0/1). In jouw df is Survived normaal al 0/1.
        if 'Survived' in existing_cols:
            if df_corr_dummied['Survived'].dtype == 'object':
                df_corr_dummied['Survived'] = df_corr_dummied['Survived'].map({'Ja': 1, 'Nee': 0})
 
        corr_matrix_nieuw = df_corr_dummied[existing_cols].corr()
 
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix_nieuw, annot=True, fmt='.2f', cmap='vlag', ax=ax_corr, vmin=-1, vmax=1)
        ax_corr.set_title('Interactieve Correlatie Matrix')
        plt.tight_layout()
        st.pyplot(fig_corr)
    else:
        st.info("Selecteer ten minste twee variabelen om een correlatiematrix te tonen.")
 
    # Reset stijl voor het geval er nog plots na komen
    sns.set_style('white')
 
 
# ===================== OUDE VISUALISATIES MET KOLONNEN =====================
else:
    st.header("Oude Visualisaties")
 
    # ===================== DATA INLADEN =====================
    train = pd.read_csv(("data/train.csv"))
    train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
    train['Age'].fillna(train['Age'].median(), inplace=True)
    if 'Cabin' in train.columns:
        train.drop(['Cabin'], axis=1, inplace=True)
 
    # Nieuwe kolommen
    fare_bins = [0, 11, 29, 70, np.inf]
    fare_labels = ['3e klasse', '2e klasse', '1e klasse (laag)', '1e klasse (£70+)']
    train['Fare_bin'] = pd.cut(train['Fare'], bins=fare_bins, labels=fare_labels)
 
    age_bins = [0, 12, 20, 40, 120]
    age_labels = ['Children', 'Teenage', 'Adult', 'Elder']
    train['Age_bin'] = pd.cut(train['Age'], bins=age_bins, labels=age_labels)
 
    train["Overleefd"] = train["Survived"].map({0: "Nee", 1: "Ja"})
    train["Geslacht"] = train["Sex"].map({'male': "Man", 'female': "Vrouw"})
 
    # ===================== TABELLEN =====================
    st.subheader("Overlevingspercentages")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Per klasse:**")
        st.write(pd.crosstab(train['Pclass'], train['Survived'], normalize='index'))
    with col2:
        st.markdown("**Per inschepingslocatie:**")
        st.write(pd.crosstab(train['Embarked'], train['Survived'], normalize='index'))
    with col3:
        st.markdown("**Per Fare_bin:**")
        st.write(pd.crosstab(train['Fare_bin'], train['Survived'], normalize='index'))
 
    # ===================== BARPLOTS =====================
    st.subheader("Overlevingspercentages per categorie")
    col1, col2, col3 = st.columns(3)
 
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x='Pclass', y='Survived', data=train, palette="pastel", ax=ax)
        ax.set_title('Overleving per klasse', fontsize=12)
        ax.set_xlabel('Klasse', fontsize=10)
        ax.set_ylabel('Gemiddeld overlevingspercentage', fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
 
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x='Embarked', y='Survived', data=train, palette="pastel", ax=ax)
        ax.set_title('Overleving per inschepingslocatie', fontsize=12)
        ax.set_xlabel('Embarked', fontsize=10)
        ax.set_ylabel('Gemiddeld overlevingspercentage', fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
 
    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(
            x='Fare_bin', y='Survived', data=train,
            order=['3e klasse', '2e klasse', '1e klasse (laag)', '1e klasse (£70+)'],
            palette="pastel", ax=ax
        )
        ax.set_title('Overleving per Fare categorie', fontsize=12)
        ax.set_xlabel('Fare categorie', fontsize=10)
        ax.set_ylabel('Gemiddeld overlevingspercentage', fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
 
    # ===================== BOXPLOT =====================
    st.subheader("Leeftijdsverdeling en overleving per klasse")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=train, x='Pclass', y='Age', hue='Overleefd', palette='pastel', ax=ax)
    sns.pointplot(data=train, x='Pclass', y='Age', hue='Overleefd',
                  dodge=0.4, join=False, markers='.', palette='dark:black',
                  errorbar=None, legend=False, ax=ax)
    ax.set_title("Leeftijdsverdeling per klasse", fontsize=14, weight="bold")
    ax.set_xlabel("Passagiersklasse", fontsize=12)
    ax.set_ylabel("Leeftijd", fontsize=12)
    fig.tight_layout()
    st.pyplot(fig)
 
    # ===================== COUNTPLOTS =====================
    st.subheader("Aantal personen per categorie en overleving")
    features = ['Geslacht', 'Pclass', 'Age_bin', 'Fare_bin']
    col_map = {
        'Geslacht': 'Geslacht',
        'Pclass': 'Passagiersklasse',
        'Age_bin': 'Leeftijdscategorie',
        'Fare_bin': 'Ticketprijs Categorie'
    }
 
    for i in range(0, len(features), 2):  # 2 plots per rij
        col1, col2 = st.columns(2)
        for j, col in enumerate(features[i:i + 2]):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x=col, hue='Overleefd', data=train, palette='pastel', ax=ax)
            ax.set_title(f"Overleving per {col_map.get(col, col)}", fontsize=12)
            ax.set_xlabel(col_map.get(col, col), fontsize=10)
            ax.set_ylabel("Aantal personen", fontsize=10)
            ax.legend(title="Overleefd", labels=['Nee', 'Ja'])
            fig.tight_layout()
            if j == 0:
                with col1:
                    st.pyplot(fig)
            else:
                with col2:
                    st.pyplot(fig)
 
    # ===================== SCATTERPLOTS =====================
    st.subheader("Relaties tussen variabelen (Scatterplots)")
 
    col1, col2 = st.columns(2)
 
    # Scatterplot 1: Leeftijd vs Fare, kleur = Overleefd
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=train,
            x='Age', y='Fare',
            hue='Overleefd',
            palette='pastel',
            alpha=0.7,
            ax=ax
        )
        ax.set_title("Leeftijd vs Ticketprijs", fontsize=12)
        ax.set_xlabel("Leeftijd", fontsize=10)
        ax.set_ylabel("Ticketprijs (£)", fontsize=10)
        ax.legend(title="Overleefd")
        fig.tight_layout()
        st.pyplot(fig)
 
    # Scatterplot 2: Leeftijd vs Fare, kleur = Geslacht + stijl = Overleefd
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=train,
            x='Age', y='Fare',
            hue='Geslacht',
            style='Overleefd',
            palette='pastel',
            alpha=0.7,
            ax=ax
        )
        ax.set_title("Leeftijd vs Ticketprijs per Geslacht & Overleving", fontsize=12)
        ax.set_xlabel("Leeftijd", fontsize=10)
        ax.set_ylabel("Ticketprijs (£)", fontsize=10)
        ax.legend(title="Geslacht / Overleefd", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        st.pyplot(fig)
 
    # ===================== AFBEELDING =====================
    img_path = "images/correlatie_heatmap_dummies.png"
    try:
        img = Image.open(img_path)
        st.subheader("Correlatieheatmap")
        st.image(img, caption="Bron: Titanic dataset", use_container_width=True)
    except Exception:
        st.info("Lokale afbeelding voor correlatieheatmap niet gevonden.")
 
