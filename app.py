from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Fetch dataset and initialize vectorizer and LSA
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000)
X = vectorizer.fit_transform(documents)

# Apply SVD for LSA
svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Vectorize and reduce the query
    query_vec = vectorizer.transform([query])
    query_reduced = svd.transform(query_vec)

    # Calculate cosine similarities between query and all documents
    similarities = cosine_similarity(query_reduced, X_reduced)[0]

    # Get the top 5 document indices
    top_indices = np.argsort(similarities)[::-1][:5]
    top_documents = [documents[i] for i in top_indices]
    top_similarities = [similarities[i] for i in top_indices]

    return top_documents, top_similarities, top_indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices})

if __name__ == '__main__':
    app.run(debug=True)
