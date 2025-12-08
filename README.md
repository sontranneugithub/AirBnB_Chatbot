# NYC Airbnb Information Retrieval Agent
# CS 4100: Son Tran, Rithvik Gowda

## Abstract

This project implements an **AI Agent system** for the NYC Airbnb dataset (2019). The agent uses a **Query Parser** to interpret user intent (e.g., price analysis, neighborhood search) and applies **TF-IDF vectorization** for semantic search over listing names. The agent dynamically generates prompts, incorporating data context, and uses the **Flan-T5** model to produce accurate, natural language answers to complex queries, such as "What is the average price in Brooklyn?" or "Show me the cheapest rooms in Manhattan."

---

## Getting Started

These instructions will guide you through setting up the environment, obtaining the necessary data, and running the `agent.ipynb` Jupyter Notebook.

### Prerequisites

You need a working Python environment (Python 3.9+) and the ability to run Jupyter Notebooks.

### Setup

1.  **Clone the Repository**

    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [YOUR_REPOSITORY_NAME]
    ```

2.  **Create Python Environment**

    The project requires a specific set of libraries, particularly `transformers` for the LLM and `scikit-learn` for TF-IDF. Use the following command to install all necessary packages via `pip`:

    ```bash
    # Install all required Python libraries
    pip install pandas numpy scikit-learn transformers jupyter joblib sqlite3
    ```

3.  **Download the Dataset**

    The agent is initialized using a specific CSV file. Download the **New York City Airbnb Open Data (2019)** and place it in the project's root directory.

    * **Data Source Link:** [New York City Airbnb Open Data (Kaggle)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
    * **Required Filename:** `AB_NYC_2019.csv`

---

## How to Run the Agent

1.  **Start Jupyter Notebook**

    ```bash
    jupyter notebook
    ```
2.  **Execute the Notebook**
    * Open `agent.ipynb` in your browser.
    * Run the cells sequentially from top to bottom.
    * The notebook performs the following steps:
        1.  Defines all core classes (`QuerySpec`, `TfidfSearch`, `PromptDB`, `LanguageModel`, `AirbnbAgent`).
        2.  Loads and cleans the `AB_NYC_2019.csv` data.
        3.  Initializes the **`TfidfSearch`** index.
        4.  Initializes the **`AirbnbAgent`** (which loads the `google/flan-t5-base` model).
        5.  Runs multiple sample queries to demonstrate the agent's various functionalities (average price, ranking, semantic search).
---
## Approach and Technical Components

The agent implements a multi-stage **Retrieve-then-Generate** workflow:

### 1. Query Parsing and Task Assignment
* **Component:** `parse_query`
* **Methodology:** Uses regular expressions to extract parameters like `borough`, `max_price`, and identifies the user's intent to determine the required task (e.g., `avg_price`, `cheapest`).

### 2. Information Retrieval (IR)
* **Component:** `TfidfSearch`
* **Methodology:** The agent combines standard data filtering (for price, borough) with **TF-IDF vectorization** and **Cosine Similarity** on the `name` (listing title) column to find listings that are semantically relevant to the user's query text.

### 3. Language Model Integration
* **Model:** **`google/flan-t5-base`**
* **Prompting:** The agent selects a specific, pre-defined template from the **`PromptDB`** based on the identified task. It injects the structured data results (the *context*) and the user's question into this template. The LLM then performs a text-to-text generation task, summarizing the provided context into a natural, helpful response.
---

## Usage Examples
You can modify the cells near the end of the `agent.ipynb` file to test custom queries:

| Query Type | Example Query |
| :--- | :--- |
| **Simple Search** | "Show me rooms with fast wi-fi" |
| **Price Analysis** | "What is the average price in Manhattan?" |
| **Budget Search** | "I need a room in Brooklyn for under $100" |
| **Ranking** | "Show me the 5 cheapest listings in Queens" |
| **High Demand** | "What neighborhoods are high in demand?" |
---

## Repository Structure

```
AirBnB_Chatbot/
├── data/
│   └── AB_NYC_2019.csv                 # The dataset used for the agent.
│
├── notebook/
│   ├── agent.ipynb                     # Initial draft or experimental notebook.
│   ├── exploration.ipynb               # Main workflow notebook for execution, testing, and logging.
│   └── prompts.db                      # Local SQLite DB used during notebook experiments.
│
├── results/
│   └── logs/
│       └── agent_run_log_YYYYMMDD_HHMMSS.json   # Time-stamped JSON logs of agent interactions.
│
├── src/                                # Source Code Directory
│   ├── __pycache__/                    # Bytecode cache for src modules
│   ├── data_processing.py              # Handles data loading, cleaning, and PromptDB initialization.
│   ├── evaluate.py                     # Query parsing, filtering logic, and statistical summaries.
│   ├── prompts.db                      # Main SQLite DB containing chatbot prompt templates.
│   └── train.py                        # Core logic: AirbnbAgent, LanguageModel, TF-IDF search, logging.
│
├── agent.py                            # Old monolithic script (deprecated; replaced by src/ modules)
│
├── environment.yml                     # Conda environment listing all project dependencies.
│
└── README.md                           # Project documentation.
```

