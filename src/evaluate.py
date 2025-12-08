from dataclasses import dataclass
from typing import Optional
import re
import pandas as pd

@dataclass
class QuerySpec:
    raw_text: str
    task: str
    borough: Optional[str] = None
    cheapest: bool = False
    most_expensive: bool = False
    avg_price: bool = False
    high_reviews_low_availability: bool = False
    max_price: Optional[float] = None
    k: int = 5


BOROUGH_ALIASES = {
    "manhattan": "Manhattan",
    "brooklyn": "Brooklyn",
    "queens": "Queens",
    "bronx": "Bronx",
    "staten island": "Staten Island",
    "staten": "Staten Island",
}


def _extract_all_numbers(text: str):
    """Extract all numbers, ignoring currency symbols."""
    cleaned = re.sub(r"[$,]", "", text)
    nums = []
    for token in cleaned.split():
        try:
            nums.append(float(token))
        except:
            pass
    return nums

def parse_query(text: str) -> QuerySpec:
    # Normalize input
    lower = text.lower()
    lower = re.sub(r"[·•\u2022\u00B7]", " ", lower)  # weird unicode dots
    lower = lower.replace("—", "-")
    lower = re.sub(r"\s+", " ", lower).strip()

    # Detect borough
    borough = None
    for key, canon in BOROUGH_ALIASES.items():
        if key in lower:
            borough = canon
            break

    # Detect top-k (e.g., find 10)
    k = 5  # default
    m = re.search(r"(?:find|top|show|give me|recommend)\s+(\d+)", lower)
    if m:
        k = int(m.group(1))

    # Cheapest + most expensive
    cheapest = any(w in lower for w in ["cheapest", "lowest price", "low price", "least expensive"])
    most_expensive = any(w in lower for w in ["most expensive", "highest price"])

    # Stats queries
    avg_price = "average price" in lower or "avg price" in lower or "mean price" in lower
    high_reviews_low_availability = (
        ("high reviews" in lower and "low availability" in lower)
        or "high demand" in lower
    )

    # Max price detection
    max_price = None

    # strong patterns: under/below/less than
    m_price = re.search(r"(?:under|below|less than)\s*\$?\s*(\d+)", lower)
    if m_price:
        max_price = float(m_price.group(1))

    # detect all other numbers
    nums = _extract_all_numbers(lower)

    for num in nums:
        # skip k numbers 
        if num == k:
            continue
        # Threshold to keep from misreading small numbers as prices
        if num >= 50:
            if not max_price:
                max_price = float(num)

    # Task type
    if avg_price or high_reviews_low_availability:
        task = "stats"
    else:
        task = "search"

    # Return parsed specification
    return QuerySpec(
        raw_text=text,
        task=task,
        borough=borough,
        cheapest=cheapest,
        most_expensive=most_expensive,
        avg_price=avg_price,
        high_reviews_low_availability=high_reviews_low_availability,
        max_price=max_price,
        k=k,
    )

def filter_by_spec(df: pd.DataFrame, spec: QuerySpec):
    sub = df
    if spec.borough:
        sub = sub[sub["neighbourhood_group"] == spec.borough]

    if spec.max_price:
        sub = sub[sub["price"] <= spec.max_price]

    return sub.copy()

def summarize_average_price(df: pd.DataFrame):
    if df.empty:
        return {"avg_price": None, "count": 0}
    
    return {
        "avg_price": float(df["price"].mean()),
        "median_price": float(df["price"].median()),
        "count": len(df),
    }

def find_cheapest(df: pd.DataFrame, k: int):
    return df.sort_values("price", ascending=True).head(k).copy()

def find_most_expensive(df: pd.DataFrame, k: int):
    return df.sort_values("price", ascending=False).head(k).copy()

def find_high_demand_neighbourhoods(df: pd.DataFrame, k: int = 5):
    grouped = (
        df.groupby(["neighbourhood_group", "neighbourhood"])
        .agg(
            avg_price=("price", "mean"),
            reviews=("number_of_reviews", "mean"),
            availability=("availability_365", "mean"),
            count=("id", "count"),
        )
        .reset_index()
    )
    grouped["demand_score"] = grouped["reviews"] / (grouped["availability"] + 1)
    return grouped.sort_values("demand_score", ascending=False).head(k)