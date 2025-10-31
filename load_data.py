import streamlit as st
import pandas as pd
import numpy as np
import os
# import json <-- Verwijderd

# --- Definieer de bestandspaden ---
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# Paden naar verwerkte bestanden
TRAIN_PROCESSED_FILE = os.path.join(DATA_DIR, "train_processed.csv")
TEST_PROCESSED_FILE = os.path.join(DATA_DIR, "test_processed.csv")

# --- Pad naar het oude notebook (Variabel verwijderd) ---

@st.cache_data
def load_raw_data():
    """Laadt de *ruwe* CSV-bestanden (voor 'voor' vergelijking)."""
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        return train_df.copy(), test_df.copy()
    except FileNotFoundError:
        st.error(f"FOUT: '{TRAIN_FILE}' of '{TEST_FILE}' niet gevonden in de map 'data/'.")
        return None, None

@st.cache_data
def get_cleaned_data():
    """
    Laadt de *vooraf verwerkte* data (gemaakt door run_preprocessing.py).
    """
    try:
        train_cleaned = pd.read_csv(TRAIN_PROCESSED_FILE)
        test_cleaned = pd.read_csv(TEST_PROCESSED_FILE)
        
        # Belangrijke check: zorg dat Survived (0.0/1.0) een int wordt
        if 'Survived' in train_cleaned:
            train_cleaned['Survived'] = train_cleaned['Survived'].astype(int)
        
        return train_cleaned.copy(), test_cleaned.copy()
    except FileNotFoundError:
        st.error(f"FOUT: '{TRAIN_PROCESSED_FILE}' of '{TEST_PROCESSED_FILE}' niet gevonden.")
        st.warning("Je moet eerst het `run_preprocessing.py` script draaien (één keer) om deze bestanden aan te maken.")
        return None, None

# --- Functies 'load_old_notebook' en 'show_notebook_cells' zijn verwijderd ---
# We hebben ze niet meer nodig omdat we de code direct plakken.

