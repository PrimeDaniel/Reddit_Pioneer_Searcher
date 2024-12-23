import praw
import os

print("now")

# Set up your credentials
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)


print(reddit.read_only)
# Output: True

def search_reddit(subreddit, query, limit=10):
    try:
        results = reddit.subreddit(subreddit).search(query, limit=limit)
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
            
    except Exception as e:
        print(f"An error occurred: {e}")

# Try with different parameters
print("Searching...")
search_reddit("all", "AI", limit=10)

print("Finished")
