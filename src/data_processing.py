import os

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import re
import sqlite3
import pickle
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sqlite3

def init_prompt_db(db_path="prompts.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        category TEXT,
        description TEXT,
        template TEXT
    )
    """)

    prompt_entries = [
        (
            "avg_price",
            "analysis",
            "Professional price analysis",
            """
You are a senior Airbnb pricing analyst.

User Question: {question}
Data Summary:
{context}

Write a polished explanation including:
- Clear takeaway
- Bold highlights
- Price insights
- 3–5 concise sentences in professional tone.
"""
        ),
        (
            "cheapest_listings",
            "search",
            "Cheapest listing explanation",
            """
Summarize these listings with:
- Bullet points
- Bold price highlights
- Neighborhood strengths

User Question: {question}
Listings:
{context}
"""
        ),
        (
            "high_demand",
            "analysis",
            "High demand area explanation",
            """
User Question: {question}
Demand Data:
{context}

Explain:
- Why demand is high
- Which neighborhoods stand out
- What the data implies.

Format in 3–4 strong sentences.
"""
        ),
        (
            "presentation_mode",
            "presentation",
            "Short slide-ready explanation",
            """
### Key Insight
**{context}**

Produce a short presentation-ready summary (2–3 sentences).
Focus on clarity and bold highlights.
"""
        )
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO prompts (name, category, description, template)
        VALUES (?, ?, ?, ?)
    """, prompt_entries)

    conn.commit()
    conn.close()

    print("✔ Prompt DB initialized:", db_path)

class PromptDB:
    def __init__(self, db_path="prompts.db"):
        self.db_path = db_path

    def get(self, name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT template FROM prompts WHERE name=?", (name,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

DATA_PATH = Path("../AirBnB_Chatbot/data/AB_NYC_2019.csv") # update as needed

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path.resolve()}")

    df = pd.read_csv(path)

    required_cols = {
        "name",
        "neighbourhood_group",
        "neighbourhood",
        "room_type",
        "price",
        "number_of_reviews",
        "availability_365",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=["name", "neighbourhood_group", "neighbourhood"]).copy()
    df = df[df["price"] > 0]
    df = df.reset_index(drop=True)

    df["text"] = (
        df["name"].fillna("") + " " +
        df["neighbourhood_group"].fillna("") + " " +
        df["neighbourhood"].fillna("") + " " +
        df["room_type"].fillna("")
    ).str.lower()

    print(f"[INFO] Loaded {len(df)} listings after cleaning.")
    
    return df