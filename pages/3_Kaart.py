import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px

# ======================
# Data inladen en opschonen
# ======================
df = pd.read_csv(r"C:\Users\niekt\Downloads\Titanic case 2.0\data\train.csv")

# Cabin droppen
df.drop(columns=['Cabin'], inplace=True)

# Embarked invullen
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Age invullen op basis van median van Sex, Pclass, Embarked
def fill_age(row):
    if pd.isna(row['Age']):
        median_age = df[
            (df['Sex'] == row['Sex']) &
            (df['Pclass'] == row['Pclass']) &
            (df['Embarked'] == row['Embarked'])
        ]['Age'].median()
        return median_age
    return row['Age']

df['Age'] = df.apply(fill_age, axis=1)

# Extra kolommen
df["Geslacht"] = df["Sex"].map({'male': "Man", 'female': "Vrouw"})
df["Pclass"] = df["Pclass"]

# ======================
# Route instellen
# ======================
def interpolate_points(points, stappen=100):
    coords = []
    for i in range(len(points) - 1):
        lon_start, lat_start = points[i]
        lon_eind, lat_eind = points[i+1]
        lons = np.linspace(lon_start, lon_eind, stappen)
        lats = np.linspace(lat_start, lat_eind, stappen)
        coords.extend(list(zip(lats, lons)))
    return coords

southampton = (-1.4044, 50.9097)
cherbourg   = (-1.62, 49.65)
queenstown  = (-8.3, 51.85)
sinking     = (-49.9, 41.7)
new_york    = (-74.0060, 40.7128)

actual_points = [southampton, cherbourg, queenstown, sinking]
planned_points = [sinking, new_york]

actual_route = interpolate_points(actual_points, stappen=200)
planned_route = interpolate_points(planned_points, stappen=100)

# Checkpoints voor havens en zinken
checkpoints = {
    "Southampton": 0,
    "Cherbourg": int(len(actual_route)*0.2),
    "Queenstown": int(len(actual_route)*0.4),
    "IJsberg": int(len(actual_route)*0.6),
    "Zinken": len(actual_route)-1
}

# ======================
# Functie om passagiers stats te berekenen
# ======================
def calc_passengers(pos):
    # Pas filter voor overlevenden toe alleen bij het allerlaatste punt
    if pos == checkpoints["Zinken"]:
        subset = df[df["Survived"]==1]
    # Accumuleer passagiers per haven
    elif pos < checkpoints["Cherbourg"]:
        subset = df[df['Embarked']=='S']  # Alleen Southampton
    elif pos < checkpoints["Queenstown"]:
        subset = df[df['Embarked'].isin(['S','C'])]  # Southampton + Cherbourg
    else:
        subset = df[df['Embarked'].isin(['S','C','Q'])]  # Alle havens

    total = len(subset)
    pct_male = (subset["Geslacht"]=="Man").sum()/total*100 if total>0 else 0
    pct_female = 100 - pct_male if total>0 else 0
    pct_class1 = (subset["Pclass"]==1).sum()/total*100 if total>0 else 0
    pct_class2 = (subset["Pclass"]==2).sum()/total*100 if total>0 else 0
    pct_class3 = (subset["Pclass"]==3).sum()/total*100 if total>0 else 0
    return total, pct_male, pct_female, pct_class1, pct_class2, pct_class3

# ======================
# Streamlit layout
# ======================
st.title("Titanic Route met Interactieve Passagiersinformatie")

st.markdown("""
**Uitleg kaart:**  
- Blauwe lijn: Werkelijke route van de Titanic  
- Gestippelde rode lijn: Geplande route naar New York  
- Groene markers: Havens  
- Rode marker: IJsberg incident  
- Grijze marker: Huidige positie van de Titanic op de route
""")

# Slider boven de kaart
route_pos = st.slider("Positie van de Titanic op de route", 0, len(actual_route)-1, 0)

# Bereken passagiers statistieken
total, pct_male, pct_female, pct_c1, pct_c2, pct_c3 = calc_passengers(route_pos)

# Alleen totaal aantal passagiers boven de kaart
st.markdown(f"**Aantal passagiers tot nu toe:** {total}")

# Kaart en diagrammen naast elkaar
col1, col2 = st.columns([2,1])

with col1:
    # Kaart
    m = folium.Map(location=[48, -40], zoom_start=3)  # Zeekaart-achtig uiterlijk
    folium.PolyLine(actual_route, color="blue", weight=3, opacity=1).add_to(m)
    folium.PolyLine(planned_route, color="red", weight=3, opacity=1, dash_array="10,10").add_to(m)

    # Havenmarkers
    stops = [
        {"name": "Southampton", "coords": [southampton[1], southampton[0]]},
        {"name": "Cherbourg",   "coords": [cherbourg[1], cherbourg[0]]},
        {"name": "Queenstown",  "coords": [queenstown[1], queenstown[0]]}
    ]
    for stop in stops:
        folium.Marker(
            stop["coords"], popup=stop["name"],
            icon=folium.Icon(color="green", icon="anchor", prefix="fa")
        ).add_to(m)

    # IJsberg incident
    folium.Marker([sinking[1], sinking[0]], popup="IJsberg incident",
                  icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa")).add_to(m)

    # Bestemming
    folium.Marker([new_york[1], new_york[0]], popup="New York",
                  icon=folium.Icon(color="blue", icon="flag")).add_to(m)

    # Bootpositie
    boot_lat, boot_lon = actual_route[route_pos]
    folium.Marker([boot_lat, boot_lon], popup=f"Titanic positie (punt {route_pos})",
                  icon=folium.Icon(color="gray", icon="ship", prefix="fa")).add_to(m)

    st_folium(m, width=900, height=500)

with col2:
    # Taartdiagram Geslacht
    fig_sex = px.pie(
        names=["Man", "Vrouw"],
        values=[pct_male, pct_female],
        title="Geslacht Passagiers",
        color_discrete_map={"Man":"#636EFA","Vrouw":"#EF553B"},
        hole=0.3
    )
    st.plotly_chart(fig_sex, use_container_width=True)

    # Taartdiagram Klasse
    fig_class = px.pie(
        names=["Klasse 1","Klasse 2","Klasse 3"],
        values=[pct_c1,pct_c2,pct_c3],
        title="Passagiers per Klasse",
        color_discrete_map={"Klasse 1":"#00CC96","Klasse 2":"#AB63FA","Klasse 3":"#FFA15A"},
        hole=0.3
    )
    st.plotly_chart(fig_class, use_container_width=True)
