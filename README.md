# ask-reddit-gpt

AskRedditGPT is a tool that takes in a query, sends it over to Reddit, and returns an answer based on data from the relevant posts and comments found in Reddit.

## Methodology

1. Take in query $q$ from user.
2. Get $N$ topics from $q$ using GPT.
3. Determine $S$, which is a set of subreddits best-suited to answer $N$ topics.
4. Search $q \in S$.
5. Retrieve a set of segments that can answer $q$.
6. Summarise segments using GPT and return answer to user.