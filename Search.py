import praw
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()
print("now")

# Set up your credentials
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

print(reddit.read_only)
# Output: True

def save_results_to_file(results, filename="results.txt"):
    with open(filename, 'w') as file:
        for post in results:
            file.write(f"Title: {post.title}\n")
            reddit_url = f"https://reddit.com{post.permalink}"
            file.write(f"Reddit Post URL: {reddit_url}\n")
            file.write(f"Score: {post.score}\n")
            file.write("-" * 50 + "\n")

def search_reddit(subreddit, query, limit=10, save=False):
    try:
        results = list(reddit.subreddit(subreddit).search(query, limit=limit))
        found_any = False
        for post in results:
            found_any = True
            print(f"\nTitle: {post.title}")
            reddit_url = f"https://reddit.com{post.permalink}"
            print(f"Reddit Post URL: {reddit_url}")
            print(f"Score: {post.score}")
            print("-" * 50)
        
        if not found_any:
            print(f"No results found for '{query}' in r/{subreddit}")
            
        if save and found_any:
            save_results_to_file(results)
        
        return [post.title for post in results]
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def calculate_tfidf(titles):
    if not titles:
        print("No titles to process for TF-IDF.")
        return
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(titles)
    
    feature_names = vectorizer.get_feature_names_out()
    for i, title in enumerate(titles):
        print(f"\nTitle: {title}")
        tfidf_scores = zip(feature_names, tfidf_matrix[i].toarray()[0])
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        for term, score in sorted_scores[:10]:  # Display top 10 terms
            print(f"Term: {term}, TF-IDF: {score}")

# Try with different parameters
print("Searching...")
titles = search_reddit("all", "My dogs love music a lot, and often listen to the Rolling Stones", limit=10)

print("Calculating TF-IDF...")
calculate_tfidf(titles)

print("Finished")
