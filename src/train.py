import pandas as pd
import numpy as np
import re

from pathlib import Path
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_processing import init_prompt_db, load_dataset, PromptDB, DATA_PATH # Import dependencies
from evaluate import (
    parse_query, filter_by_spec, summarize_average_price, 
    find_cheapest, find_most_expensive, find_high_demand_neighbourhoods
)

class TfidfSearch:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(df["text"].fillna(""))

    def search(self, query: str, k: int = 10, candidate_idx=None):
        if candidate_idx is None:
            sub_matrix = self.matrix
            base_df = self.df
        else:
            valid_idx = candidate_idx[candidate_idx < self.matrix.shape[0]]
            sub_matrix = self.matrix[valid_idx]
            base_df = self.df.iloc[valid_idx]


        q_vec = self.vectorizer.transform([query.lower()])
        sims = cosine_similarity(q_vec, sub_matrix)[0]

        order = np.argsort(sims)[::-1][:k]
        result = base_df.iloc[order].copy()
        result["similarity"] = sims[order]
        return result


class LanguageModel:
    def __init__(self, model_name="google/flan-t5-base"):
        self.pipe = None
        if pipeline is None:
            print("[WARN] transformers not available. Using fallback mode.")
        else:
            try:
                self.pipe = pipeline("text2text-generation", model=model_name)
            except Exception as e:
                print(f"[WARN] Model load failed: {e}")
                self.pipe = None

    def generate(self, prompt: str, max_new_tokens=128):
        if self.pipe is None:
            return "[No LLM available] " + prompt
        out = self.pipe(prompt, max_length=max_new_tokens, num_beams=4)[0]["generated_text"]
        return out.strip()

class AirbnbAgent:
    def __init__(self, df):
        self.df = df
        self.search_index = TfidfSearch(df)
        self.lm = LanguageModel()
        self.promptdb = PromptDB()  


    def _format_rows(self, rows, max_rows=10):
        lines = []
        for _, r in rows.head(max_rows).iterrows():
            lines.append(
                f"- {r['neighbourhood_group']} / {r['neighbourhood']} | "
                f"{r['room_type']} | ${r['price']} | "
                f"reviews={r['number_of_reviews']} | avail={r['availability_365']}"
            )
        return "\n".join(lines) or "No listings matched."

    def answer(self, question: str) -> str:
        clean_q = question
        clean_q = re.sub(r"[·•\u2022\u00B7]", " ", clean_q)  
        clean_q = clean_q.replace("—", "-")                 
        clean_q = re.sub(r"\s+", " ", clean_q).strip()     

        clean_q
        
        spec = parse_query(clean_q)
        sub = filter_by_spec(self.df, spec)

        if sub.empty:
            return "No listings matched your query."
    
        # Stats: average price
        if spec.avg_price:
            stats = summarize_average_price(sub)

            context = (
                f"Average price ≈ ${stats['avg_price']:.0f}, "
                f"median = ${stats['median_price']:.0f}, "
                f"based on {stats['count']} listings."
            )

            template = self.promptdb.get("avg_price")   # <-- SQL prompt
            prompt = template.format(question=clean_q, context=context)
            return self.lm.generate(prompt)

        # High demand areas (reviews high, availability low)
        if spec.high_reviews_low_availability:
            demand = find_high_demand_neighbourhoods(sub, spec.k)
            context = demand.to_string(index=False)

            template = self.promptdb.get("high_demand")   # <-- SQL prompt
            prompt = template.format(question=clean_q, context=context)
            return self.lm.generate(prompt)

        # Search queries + ranking
        candidate_idx = sub.index.to_numpy()
        ranked = self.search_index.search(clean_q, k=spec.k, candidate_idx=candidate_idx)

        if spec.cheapest:
           ranked = find_cheapest(sub, spec.k)
           prompt_name = "cheapest_listings"
        elif spec.most_expensive:
            ranked = find_most_expensive(sub, spec.k)
            prompt_name = "cheapest_listings"  # same template works well
        else:
            prompt_name = "cheapest_listings"
        
        context = self._format_rows(ranked, spec.k)

    # Load SQL template
        template = self.promptdb.get(prompt_name)
        prompt = template.format(question=clean_q, context=context)
        return self.lm.generate(prompt)

# --- src/train.py (LogRecorder Class) ---

import json
import numpy as np
from pathlib import Path
from datetime import datetime

class LogRecorder:
    def __init__(self, base_dir="results/logs"):
        # Resolves the path relative to the project root (../)
        self.base_dir = Path("../") / base_dir 
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[LOG] Log directory established at: {self.base_dir.resolve()}")

    def record_run(self, log_records: list):
        """
        Saves a list of structured log dictionaries to a time-stamped JSON file.
        This method embodies the software approach by handling path management
        and serialization complexity.
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"agent_run_log_{timestamp}.json"
        log_path = self.base_dir / log_file_name

        def json_default_serializer(obj):
            """Custom function to serialize non-standard data types."""
            if isinstance(obj, Path):
                return str(obj)
            # CRITICAL: Converts NumPy types (used by Pandas/DataFrames) to standard Python types
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        print(f"[LOG] Writing {len(log_records)} records to {log_path.resolve()}")
        
        try:
            with open(log_path, 'w') as f:
                # The 'default' argument ensures complex types are handled by the custom serializer
                json.dump(log_records, f, indent=4, default=json_default_serializer)
            print("✔ Log export complete.")
        except Exception as e:
            print(f"[ERROR] Failed to write log: {e}")
        
        return log_path

# Execution
if __name__ == '__main__':
    # setup and initialization
    init_prompt_db() 
    df = load_dataset(DATA_PATH)
    agent = AirbnbAgent(df)
    recorder = LogRecorder()
    log_records = []

    # run agent on sample queries
    queries = [
        "What is the average price in Manhattan for a private room under $150?",
        "Show me the 5 cheapest listings in Brooklyn that have good reviews.",
        "Where are the highest demand areas in Queens?"
    ]
    
    print("\n--- Running Agent Queries ---")
    for q in queries:
        
        # Get the result
        raw_result = agent.answer(q) 
        
        if isinstance(raw_result, dict):
            # Success Case: The agent returned a structured log entry
            log_entry = raw_result
            log_records.append(log_entry)
            response_text = log_entry['final_response']
        elif isinstance(raw_result, str):
            # Failure Case: The agent returned the raw string 
            log_entry = {
                "query": q,
                "parsed_spec": {"error": "Agent returned raw string, not dictionary."},
                "final_response": raw_result
            }
            log_records.append(log_entry)
            response_text = raw_result
        else:
            # Unknown failure
            response_text = f"Agent returned an unhandled type: {type(raw_result).__name__}"
            log_records.append({"query": q, "error": response_text})

        print(f"\nQ: {q}")
        print(f"A: {response_text}") 
        print("-" * 50)

    # Record results
    recorder.record_run(log_records)