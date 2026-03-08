# Semantic Search System with Fuzzy Clustering and Semantic Cache

This project implements a **semantic document search system** capable of understanding the **meaning of user queries** instead of relying on traditional keyword-based search.

The system uses **transformer-based embeddings, vector similarity search, fuzzy clustering, and a custom semantic caching mechanism** to efficiently retrieve relevant documents.

This project was developed as part of an **AI/ML Engineering task submission**.

---

# Project Overview

Traditional search engines rely heavily on **keyword matching**, which often fails to capture the true intent behind a user query.

This project demonstrates how **modern NLP techniques** can be used to build a more intelligent search system.

The system:

- Converts documents into **semantic embeddings**
- Stores embeddings in a **vector database**
- Uses **fuzzy clustering** to model topic relationships
- Implements a **semantic cache** to improve performance for repeated queries

The system is tested using the **20 Newsgroups Dataset**, which contains **20,000 discussion posts across 20 different topics**.

---

# Key Features

## 1. Semantic Search

Instead of exact keyword matching, the system retrieves documents based on **semantic similarity**.

Example query:
computer graphics algorithms


The system can return documents related to:

- GPU rendering
- visualization techniques
- graphics processing

even if the exact keywords do not appear in the documents.

---

## 2. Transformer-Based Embeddings

All documents and queries are converted into **vector embeddings** using transformer models.

These embeddings capture the **context and meaning** of text, enabling accurate similarity comparisons.

---

## 3. Vector Database Search

Document embeddings are stored using **FAISS**, which allows extremely fast **nearest-neighbor search** across thousands of vectors.

This enables scalable semantic search.

---

## 4. Fuzzy Clustering

Instead of assigning documents to only one topic, the system uses **Gaussian Mixture Model clustering** to implement **soft cluster membership**.

Example:
Document: Space Technology

Cluster Membership:
Space : 0.68
Technology : 0.22
Science : 0.10


This allows documents to belong to **multiple topics simultaneously**.

---

## 5. Custom Semantic Cache

The project includes a **semantic caching system built from scratch**.

When a query is received:

1. The system computes the query embedding
2. It compares it with embeddings of previously asked queries
3. If similarity exceeds a threshold → cached results are returned
4. Otherwise → a new search is executed and stored in cache

The cache stores:

- query text
- query embedding
- retrieved documents
- dominant cluster
- timestamp

This significantly improves performance for repeated or similar queries.

---

# Example response:

{
 "query": "computer graphics algorithms",
 "cache_hit": false,
 "similarity_score": 0.89,
 "dominant_cluster": 4,
 "documents": [...]
}

# Cache Statistics
GET /cache/stats

Returns:

total cache entries

cache hits

cache misses

cache hit rate

Clear Cache
# DELETE /cache

Clears all cached queries.

# Installation

Clone the repository:

git clone https://github.com/yourusername/semantic-search-system.git

Navigate to the project directory:

cd semantic-search-system

Install dependencies:

pip install -r requirements.txt

Run the FastAPI server:

uvicorn app.main:app --reload

The API will start at:

http://127.0.0.1:8000
