import streamlit as st
st.set_page_config(page_title="Conclusie", layout="wide", page_icon="🏁")


st.title("🏁 Conclusie Correlatieanalyse")

st.header("Antwoord op de Onderzoeksvraag")

st.markdown("""
De hoofdvraag van dit project was: **"Welke factoren bepaalden de overlevingskans van een passagier?"**


Geslacht en de daarvan afgeleide Titel tonen het sterkste statistische verband.
Daarnaast correleren passagiersklasse en de bijbehorende ticketprijs significant met de kans op overleving.
Ook de familiegrootte vertoont een duidelijk verband; alleen reizen correleert negatief met overleving, terwijl reizen in een kleine groep positief correleert.
Deze factoren samen geven een duidelijk beeld van de indicatoren die samenhangen met de overlevingskans.
""")
# --- EINDE AANPASSING ---