from langchain.embeddings import TensorflowHubEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from tqdm import tqdm
import pandas as pd
import gradio as gr
import openai
import praw
import os
import re

reddit = None
bot = None
chat_history = []

def set_openai_key(key):

    if key == "":
        key = os.environ.get("OPENAI_API_KEY")

    openai.api_key = key

def set_reddit_keys(client_id, client_secret, user_agent):

    global reddit

    # If any of the keys are empty, use the environment variables
    if [client_id, client_secret, user_agent] == ["", "", ""]:
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        user_agent = os.environ.get("REDDIT_USER_AGENT")

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

def generate_topics(query, model="gpt-3.5-turbo"):

    messages = [
        {"role": "user", "content": f"Take this query '{query}' and return a list of 10 simple to understand topics (4 words or less) to input in Search so it returns good results."},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )

    response_message = response["choices"][0]["message"]["content"]

    topics = re.sub(r'^\d+\.\s*', '', response_message, flags=re.MULTILINE).split("\n")

    # Post-processing GPT output

    topics = [topic.strip() for topic in topics]

    topics = [topic[1:-1] if (topic.startswith('"') and topic.endswith('"')) or (topic.startswith("'") and topic.endswith("'")) else topic for topic in topics]

    topics = [re.sub(r'[^a-zA-Z0-9\s]', ' ', topic) for topic in topics]

    return topics

def get_relevant_comments(topics):

    global reddit

    comments = []

    for topic in tqdm(topics):
        for post in reddit.subreddit("all").search(
        topic, limit=10):
            
            post.comment_limit = 20
            post.comment_sort = "top"

            # Top level comments only
            post.comments.replace_more(limit=0)

            for comment in post.comments:
                author = comment.author.name if comment.author else '[deleted]'
                comments.append([post.id, comment.id, post.subreddit.display_name, post.title, author, comment.body])

    comments = pd.DataFrame(comments,columns=['source', 'comment_id', 'subreddit', 'title', 'author', 'text'])

    # Drop empty texts or ["deleted"] texts
    comments = comments[comments['text'].str.len() > 0]
    comments = comments[comments['text'] != "[deleted]"]

    # Drop comments with None authors
    comments = comments[comments['author'] != "AutoModerator"]

    # Drop duplicate ids
    comments = comments.drop_duplicates(subset=['source'])

    return comments

def construct_retriever(comments, k=20):

    # Convert comments dataframe to a dictionary
    comments = comments.to_dict('records')

    # Convert comments["text"] to a list of strings
    texts = [comment["title"] + " " + comment["text"] + " " + comment["subreddit"] for comment in comments]

    db = Chroma.from_texts(texts, TensorflowHubEmbeddings(model_url="https://tfhub.dev/google/universal-sentence-encoder/4"), metadatas=[{"source": comment["source"], "comment_id": comment["comment_id"], "author": comment["author"]} for comment in comments])

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    return retriever

def construct_bot(retriever):
    bot = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever, return_source_documents=True)
    return bot

def get_response(query, chat_history):
    response = bot({"question": query, "chat_history": chat_history})
    return response

def restart():
    
    global chat_history
    global bot

    chat_history = []
    bot = None

    print("Chat history and bot knowledge has been cleared!")

    return None

def main(query):

    global chat_history
    global bot

    if chat_history == []:
        print("Bot knowledge has not been initialised yet! Generating topics...")
        topics = generate_topics(query)

        print("Fetching relevant comments...")
        comments = get_relevant_comments(topics)

        print("Embedding relevant comments...")
        retriever = construct_retriever(comments)

        print("Educating bot...")
        bot = construct_bot(retriever)

        print("Bot has been constructed and is ready to use!")

    response = get_response(query, chat_history)

    answer, source_documents = response["answer"], response["source_documents"]

    print(source_documents)

    chat_history.append((query, answer))

    return "", chat_history

# Testing only!
set_openai_key("")
set_reddit_keys("", "", "")

with gr.Blocks() as demo:
    chat_bot = gr.Chatbot()
    query = gr.Textbox()
    clear = gr.Button("Clear")

    query.submit(main, [query], [query, chat_bot])
    clear.click(restart, None, chat_bot, queue=False)

demo.launch()