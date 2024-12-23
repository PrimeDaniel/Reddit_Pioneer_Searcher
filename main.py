import praw

print("fdgdff")

# Set up your credentials
reddit = praw.Reddit(
    client_id="nmcA32g8dnE_Dc3cBfr38Q",
    client_secret="ECvq3EUxRiJ7P3F587RLoCmdkCwncw",
    user_agent="my_reddit_search/1.0/Precise_Pioneer"
)

# Search posts
def search_reddit(subreddit, query, limit=10):
    results = reddit.subreddit(subreddit).search(query, limit=limit)
    for post in results:
        print(f"Title: {post.title}, URL: {post.url}, Score: {post.score}")

# Example usage
search_reddit("all", "AI", limit=5)


print("fdgdff")