# AI-Powered Ticket Classification System 🚀

This project solves the Internshala AI Assessment by classifying and grouping customer support tickets using Natural Language Processing (NLP) and Machine Learning. 

## 🧠 Assessment Approach

Given the 3 tickets:
1. "I forgot my password, how to reset it?"
2. "I can't log in, as password is incorrect."
3. "How to see leave balance?"

### How are they grouped?
The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for vectorization and **Agglomerative Hierarchical Clustering** (with Cosine Similarity) to group tickets:

*   **Group 1: Password & Authentication** (Tickets 1 & 2)
    *   Both tickets share context around "password", "reset", "login". AI successfully identifies the high cosine similarity between them.
*   **Group 2: HR & Leave Management** (Ticket 3)
    *   This ticket's intent is about "leave" and "balance", which has 0 similarity to Group 1, making it a distinct cluster.

### System Architecture
1.  **Text Preprocessing Pipeline:** Lowercasing, punctuation removal, stopword removal.
2.  **Feature Extraction Engine:** TF-IDF vectorizer (Unigrams & Bigrams).
3.  **Ticket Clusterer:** Hierarchical Clustering using Cosine distance.
4.  **Intent Classifier:** Categorizes tickets using a weighted knowledge base.
5.  **Response Generator:** Context-aware automated responses for each category.

## 🛠️ Project Structure
*   `main.py` - Core AI engine + pipeline orchestration + beautiful console output.
*   `dashboard.html` - Stunning visual dashboard generated dynamically.
*   `requirements.txt` - Dependencies (`numpy`, `scikit-learn`).

## 🚀 How to Run (For HR and Reviewers)

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Execute the AI Pipeline**:
    ```bash
    python main.py
    ```

3.  **View Dashboard**: Double-click `dashboard.html` in your file explorer to see the high-end presentation of the results!
