import praw
import os
import re
import pandas as pd
from dotenv import load_dotenv
from nltk.stem import PorterStemmer

# Load environment variables
load_dotenv()

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

print("Reddit API initialized. Read-only mode:", reddit.read_only)


# Initialize Porter Stemmer
stemmer = PorterStemmer()

# Define stopwords
STOPWORDS = set([
    "and", "or", "the", "is", "in", "to", "a", "of", "on", "for", "with", "it", "as", "at", "this", "that",
    "an", "be", "are", "by", "was", "were", "from", "has", "have", "had", "but", "not", "you", "we", "they", "he", "she", "i", "me", "my"
])


# Function to clean text- remove special characters, convert to lowercase, remove stopwords, and stem words
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    filtered_words = [stemmer.stem(word) for word in words if word not in STOPWORDS]
    return " ".join(filtered_words)

# PageRank-like function to rank posts
def apply_pagerank(df, damping=0.85, iterations=10):
    # Initialize equal ranks for all posts
    df["pagerank"] = 1 / len(df)
    for _ in range(iterations):
        new_ranks = []
        for _, post in df.iterrows():
            rank_sum = 0
            # Calculate the rank contribution from other posts
            for _, other_post in df.iterrows():
                if other_post["Subreddit"] == post["Subreddit"] and other_post["Title"] != post["Title"]:
                    rank_sum += other_post["pagerank"] / len(df[df["Subreddit"] == post["Subreddit"]])
            new_rank = (1 - damping) / len(df) + damping * rank_sum
            new_ranks.append(new_rank)
        df["pagerank"] = new_ranks
    return df

# Function to search Reddit and save results to Excel
def search_reddit_to_excel(subreddit, query, limit=20, output_file="reddit_results.xlsx"):
    try:
        print(f"Searching Reddit for '{query}' in r/{subreddit}...")
        results = reddit.subreddit(subreddit).search(query, limit=limit, sort='controversial', time_filter='day')

        data = []
        for post in results:
            data.append({
                "Title": clean_text(post.title),
                "Body": clean_text(post.selftext),  # Include cleaned post body
                "Reddit Post URL": f"https://reddit.com{post.permalink}",
                "Upvotes": post.score,
                "Comments": post.num_comments,  # Add the number of comments
                "Subreddit": post.subreddit.display_name,
            })

        if data:
            # Create a DataFrame and apply PageRank
            df = pd.DataFrame(data)
            df = apply_pagerank(df)
            
            # Sort by PageRank
            df = df.sort_values("pagerank", ascending=False)

            # Save to Excel
            df.to_excel(output_file, index=False)
            print(f"Results saved to {output_file}")
        else:
            print(f"No results found for '{query}' in r/{subreddit}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

# Main script
if __name__ == "__main__":
    try:
        search_reddit_to_excel("all", "Funny cats", limit=20)
    except Exception as e:
        print(f"Error during Reddit search: {e}")
