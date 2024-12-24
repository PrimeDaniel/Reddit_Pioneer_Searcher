import praw
import os
from dotenv import load_dotenv

load_dotenv()
print("now")

# Set up your credentials
reddit = praw.Reddit(
    client_id = os.getenv("REDDIT_CLIENT_ID"),
    client_secret = os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent = os.getenv("REDDIT_USER_AGENT"),
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
            
    except Exception as e:
        print(f"An error occurred: {e}")

# Try with different parameters
print("Searching...")
search_reddit("all", "My dogs love music a lot, and often listen to the Rolling Stones", limit=10)

print("Finished")
