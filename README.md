# NYC Airbnb Information Retrieval Agent
# CS 4100: Son Tran, Rithvik Gowda

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
├── environment.yml                     # Conda environment listing all project dependencies.
│
└── README.md                           # Project documentation.
```

## Project Documentation

### Abstract

This project implements an **AI Agent system** for the NYC Airbnb dataset (2019). The agent uses a **Query Parser** to interpret user intent (e.g., price analysis, neighborhood search) and applies **TF-IDF vectorization** for semantic search over listing names. The agent dynamically generates prompts, incorporating data context, and uses the **Flan-T5** model to produce accurate, natural language answers to complex queries, such as "What is the average price in Brooklyn?" or "Show me the cheapest rooms in Manhattan."

### Overview

Trying to find vacation rental housing can be expensive and exhausting. Platforms like Airbnb don't allow for users to just search listings based on their request. This can end up with a lot of time spent manually adjusting filters and missing out on choosing the most optimal rental for you.

This problem is interesting because it contributes to a bigger trend going on currently, where consumers want platforms they use to be more helpful and easier to navigate. There's a demand for AI systems in everyday tasks, and this demand needs to be met in the trip planning category, too.

First, we parse the question, then using TF-IDF, we search the NYC Airbnb dataset for relevant listings, and then use a small language model to generate a summary. This approach makes sense for this problem since we have short and relevant data, the listing descriptions are small, and the dataset isn't very big, so we don't need any bigger tools.

The inspiration for this is that it's kind of like a wrapper. There are already approaches to solving this problem, such as travel companies using diverse and intricate systems that run into a lot of issues. On our side, our version is accessible and easy to use, and basic enough for consumers to be happy with it.

Key components include the NYC Airbnb dataset, TF-IDF search, Regex query parsing, prompt templates, HuggingFace language model, and an agent class. Limitations include that the dataset is old, as it's from 2019, the model won't know what listings are available currently, queries and prompts file is small.

### Approach

The user first asks a question to the agent, then the agent filters the results based on the question and returns the relevant listings. TF-IDF ranks the listings based on which one is more relevant to the question, and then ranks the listings based on that. The prompt template gets picked and added with the information of the question, and the results, and then the language model makes a summary based on this.

Algorithms/models/methods used include:
Regex, TF-IDF, Cosine similarity, Prompt templates, and HuggingFace text generation.

We used TF-IDF since the descriptions for the listing are not that long. Updating prompts is easy, as they are stored in a database. Using a small model keeps the results fast and comprehensive.

Limitations include that this only deals with the 2019 NYC dataset, not very advanced compared to modern AI models; prompts and queries are limited, it doesn't allow for much diversity, and we tried implementing a memory feature that had some bugs and did not give us the result we were hoping for.

### Experiments

The dataset includes 48,895 Airbnb listings in New York from 2019 which has columns that list the neighborhood, room type, price, reviews, neighborhood groups, and much more.

We used TF-IDF with 1-gram and 2-gram features, we multiplied the Top-k by 5 for more diversity with the results, the HuggingFace text-to-text model for generating a summary, the needed dependencies in environment.yml, and the code runs in Python and can be run from your local machine.

We did not use a neural network, as we used TF-IDF for our search model and a sequence-to-sequence text generation model; the pipeline is easy to recreate.

### Results

The model was able to correctly handle price-related prompts, area-related prompts, the determination of which filters to apply, and variety. The prompts and queries may be limited, but it correctly handles the most popular questions when searching for a vacation rental.

There were a lot of parameter choices made, such as using a bigram TF-IDF as it retrieves more relevant listings and can capture and context phrases or names. We almost multiplied the Top k value by 5 in order to create more diversity in the results returned. We also got rid of outliers in the data that could mess up the model's return answer relevance. We also used prompt routing, where if keywords were picked up like “neighborhood,” it would use the corresponding template.

### Discussion

The results are strong and are the best they can be for the limits in the small generation model and TF-IDF search. The model shows that it can handle a variety of queries and prompts relating all the way from price to neighborhood. For simple queries, the model correctly extracts features and constraints and will return relevant listings. Even a step above that, it can handle some complex queries that require multiple feature filtering and ranking.

Results are strong for what our limitations were, which included the small hugging face model for generation, as well as using TF-IDF for search. This includes a limited ability to only scavenge through that given dataset and not have real-time updates on pricing and availability. Many travel companies might use more advanced tech that includes large models or embeddings. Our results are strong and relevant; however, sometimes they struggle with broad questions and repeat listing details now and then due to limitations and bugs of a low-powered generative model.

In the future, to expand this design, having an embedding-based system for retrieving would be a lot more valuable and give the model more understanding. Having a real-time API would also allow the model to have access and use live data, which would significantly increase the value and productivity of this agent.

### Conclusion

Overall, this project successfully builds a relevant AI Agent that can interpret and understand basic Airbnb queries to make vacation rental search a lot easier and faster. By using TF-IDF, strict parsing, prompt templates, and a small language model, the project replicates a full end-to-end workflow that solves a very big and relevant problem that big platforms like Airbnb have not yet addressed. The model shows that it can handle all different types of queries, from statistical queries to neighborhood recommendations to feature filtering. Finally, the project shows a great understanding of retrieval-based agent systems and will provide a foundation for a more complex travel assistant.

### References

Kaggle Dataset:
https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

Transformers Document (Hugging Face):
https://huggingface.co/docs/transformers/index

TF-IDF Vectorizer:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

SQLite Documentation:
https://www.sqlite.org/docs.html


