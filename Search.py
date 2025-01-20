import praw
import os
import re
import pandas as pd
from dotenv import load_dotenv
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# Define stopwords and lemmatizer
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in STOPWORDS]
    return " ".join(filtered_words)

# Function to search Reddit and save results to Excel
def search_reddit_to_excel(subreddit, query, limit=20, output_file="reddit_results.xlsx"):
    results = reddit.subreddit(subreddit).search(
        query=query, limit=limit, sort="hot", time_filter="day"
    )
    data = []
    for post in results:
        post_time = pd.to_datetime(post.created_utc, unit="s")  # Convert to datetime
        data.append({
            "Title": post.title,
            "Body": post.selftext,
            "Upvotes": post.score,
            "Comments": post.num_comments,
            "Subreddit": post.subreddit.display_name,
            "Post Time (UTC)": post_time,
        })
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

# Function to generate inverted index
def generate_inverted_index(input_file="reddit_results.xlsx", output_file="inverted_index.txt"):
    # Load data
    df = pd.read_excel(input_file)

    # Combine title and body into a single content column
    df["Content"] = (df["Title"].astype(str) + " " + df["Body"].fillna("")).apply(clean_text)

    # Build the inverted index
    inverted_index = defaultdict(list)
    for idx, content in enumerate(df["Content"]):
        post_id = str(idx + 1)  # Use numeric post ID starting from 1
        words = set(content.split())  # Use set to avoid duplicates
        for word in words:
            inverted_index[word].append(post_id)

    # Save inverted index to a text file
    with open(output_file, "w") as f:
        for word, post_ids in sorted(inverted_index.items()):
            f.write(f"{word}\t{', '.join(post_ids)}\n")

    print(f"Inverted index saved to {output_file}")

# Main script
if __name__ == "__main__":
    # Step 1: Search Reddit and save results
    search_reddit_to_excel("all", "Funny cats", limit=10)

    # Step 2: Generate inverted index
    generate_inverted_index()
