# app.py
from flask import Flask, render_template, jsonify, request
import praw
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
load_dotenv()

# Set up Reddit credentials
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    subreddit = data.get('subreddit', 'all')
    query = data.get('query', '')
    limit = int(data.get('limit', 10))
    save = data.get('save', False)
    
    try:
        results = list(reddit.subreddit(subreddit).search(query, limit=limit))
        posts = []
        
        for post in results:
            posts.append({
                'title': post.title,
                'url': f"https://reddit.com{post.permalink}",
                'score': post.score
            })
            
        if save:
            save_results_to_file(results)
            
        # Calculate TF-IDF
        if posts:
            titles = [post['title'] for post in posts]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(titles)
            feature_names = vectorizer.get_feature_names_out()
            
            tfidf_results = []
            for i, title in enumerate(titles):
                tfidf_scores = zip(feature_names, tfidf_matrix[i].toarray()[0])
                sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
                tfidf_results.append({
                    'title': title,
                    'terms': [{
                        'term': term,
                        'score': float(score)
                    } for term, score in sorted_scores[:10]]
                })
        
        return jsonify({
            'success': True,
            'posts': posts,
            'tfidf_analysis': tfidf_results if posts else []
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def save_results_to_file(results, filename="results.txt"):
    with open(filename, 'w') as file:
        for post in results:
            file.write(f"Title: {post.title}\n")
            reddit_url = f"https://reddit.com{post.permalink}"
            file.write(f"Reddit Post URL: {reddit_url}\n")
            file.write(f"Score: {post.score}\n")
            file.write("-" * 50 + "\n")

if __name__ == '__main__':
    app.run(debug=True)
