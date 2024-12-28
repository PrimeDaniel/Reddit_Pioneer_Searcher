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

# Define stopwords
STOPWORDS = set([
    "and", "or", "the", "is", "in", "to", "a", "of", "on", "for", "with", "it", "as", "at", "this", "that",
    "an", "be", "are", "by", "was", "were", "from", "has", "have", "had", "but", "not", "you", "we", "they", "he", "she", "i", "me", "my"
])
# Define stemmer
stemmer = PorterStemmer()

# Function to clean text- remove special characters, convert to lowercase, remove stopwords, and stem words
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    filtered_words = [stemmer.stem(word) for word in words if word not in STOPWORDS]
    return " ".join(filtered_words)

# PageRank-like function to rank posts
def apply_pagerank(df, damping=0.85, iterations=10):
    # Normalize engagement metrics
    max_upvotes = df['Upvotes'].max() if df['Upvotes'].max() != 0 else 1
    max_comments = df['Comments'].max() if df['Comments'].max() != 0 else 1
    
    # Calculate initial rank based on normalized engagement
    df['engagement_score'] = (
        (df['Upvotes'] / max_upvotes * 0.7) +  # Weight upvotes more
        (df['Comments'] / max_comments * 0.3)   # Weight comments less
    )
    
    # Initialize ranks based on engagement
    df["pagerank"] = df['engagement_score'] / df['engagement_score'].sum()
    
    for _ in range(iterations):
        new_ranks = []
        for _, post in df.iterrows():
            rank_sum = 0
            
            # Calculate similarity with other posts
            for _, other_post in df.iterrows():
                if post.name != other_post.name:  # Don't compare post with itself
                    # Calculate content similarity
                    title_similarity = len(set(post['Cleaned_Title'].split()) & 
                                        set(other_post['Cleaned_Title'].split())) > 0
                    
                    # Consider both subreddit relationship and content similarity
                    if (other_post["Subreddit"] == post["Subreddit"] or title_similarity):
                        # Weight by engagement score
                        contribution = (other_post["pagerank"] * 
                                     other_post['engagement_score']) / len(df)
                        rank_sum += contribution
            
            # Calculate new rank incorporating engagement
            new_rank = ((1 - damping) * post['engagement_score'] / 
                       df['engagement_score'].sum() + 
                       damping * rank_sum)
            new_ranks.append(new_rank)
        
        # Update ranks
        df["pagerank"] = new_ranks
        
        # Normalize ranks
        df["pagerank"] = df["pagerank"] / df["pagerank"].sum()
    
    return df

# Function to search Reddit and save results to Excel
def search_reddit_to_excel(subreddit, query, limit=20, output_file="reddit_results.xlsx"):
    try:
        print(f"\nSearching Reddit for '{query}' in r/{subreddit}...")
        results = reddit.subreddit(subreddit).search(query, limit=limit, sort='controversial', time_filter='day')

        data = []
        for post in results:
            data.append({
                "Title": post.title,
                "Body": post.selftext,
                "Post URL": f"https://reddit.com{post.permalink}",
                "Upvotes": post.score,
                "Comments": post.num_comments,  # Add the number of comments
                "Subreddit": post.subreddit.display_name,
                "Cleaned_Title": clean_text(post.title),
                "Cleaned_Body": clean_text(post.selftext),  # Include cleaned post body
                
            })

        if data:
            # Create a DataFrame and apply PageRank
            df = pd.DataFrame(data)
            df = apply_pagerank(df)
            
            # Sort by PageRank
            df = df.sort_values("pagerank", ascending=False)

            # Save to Excel
            df.to_excel(output_file, index=False)
            print(f"\nResults saved to {output_file}\n")
        else:
            print(f"\nNo results found for '{query}' in r/{subreddit}\n")
    
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise

# Main script
if __name__ == "__main__":
    try:
        search_reddit_to_excel("all", "Funny cats", limit=20)
    except Exception as e:
        print(f"Error during Reddit search: {e}")