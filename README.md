# ask-reddit-gpt

AskRedditGPT is a tool that takes in a query, sends it over to Reddit, and returns an answer based on relevant posts/comments.

## Methodology

1. Take in query $q$ from user.
2. Get $N$ topics from $q$ using GPT.
3. Determine $C$, which is a set of comments best-suited to answer $N$ topics.
4. Search $q \in C$.
5. Use GPT to return an answer to user.