from langchain.embeddings import TensorflowHubEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from tqdm import tqdm
import pandas as pd
import gradio as gr
import datetime
import openai
import praw
import os
import re

embs = TensorflowHubEmbeddings(model_url="https://tfhub.dev/google/universal-sentence-encoder/4")

def set_openai_key(key):

    if key == "":
        key = os.environ.get("OPENAI_API_KEY")

    openai.api_key = key

def set_reddit_keys(client_id, client_secret, user_agent):

    # If any of the keys are empty, use the environment variables
    if [client_id, client_secret, user_agent] == ["", "", ""]:
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        user_agent = os.environ.get("REDDIT_USER_AGENT")

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    return reddit

def generate_topics(query, model="gpt-3.5-turbo"):

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    messages = [
        {"role": "user", "content": f"The current date is {current_date}. Take this query '{query}' and return a list of 10 simple to understand topics (4 words or less) to input in Search so it returns good results."}
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

def get_relevant_comments(reddit, topics):

    comments = []

    for topic in tqdm(topics):
        for post in reddit.subreddit("all").search(
        topic, limit=5):
            
            post.comment_limit = 10
            post.comment_sort = "top"

            # Top level comments only
            post.comments.replace_more(limit=0)

            for comment in post.comments:
                author = comment.author.name if comment.author else '[deleted]'
                comments.append([post.id, comment.id, post.subreddit.display_name, post.title, author, comment.body, datetime.datetime.fromtimestamp(comment.created).strftime('%Y-%m')])

    comments = pd.DataFrame(comments,columns=['source', 'comment_id', 'subreddit', 'title', 'author', 'text', 'date'])

    # Drop empty texts or ["deleted"] texts
    comments = comments[comments['text'].str.len() > 0]
    comments = comments[comments['text'] != "[deleted]"]

    # Drop comments with None authors
    comments = comments[comments['author'] != "AutoModerator"]

    # Drop duplicate ids
    comments = comments.drop_duplicates(subset=['source'])

    return comments

def construct_retriever(comments, k=5):

    # Convert comments dataframe to a dictionary
    comments = comments.to_dict('records')

    # Convert comments["text"] to a list of strings
    texts = [comment["title"] + " " + comment["date"] + ": " + comment["text"] + " " + comment["subreddit"] for comment in comments]

    db = Chroma.from_texts(texts, embs, metadatas=[{"source": comment["source"], "comment_id": comment["comment_id"], "author": comment["author"], "subreddit": comment["subreddit"], "title": comment["title"]} for comment in comments])

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    return retriever

def construct_bot(retriever):
    bot = ConversationalRetrievalChain.from_llm(OpenAI(openai_api_key=openai.api_key, temperature=0), retriever, return_source_documents=True, max_tokens_limit=2000)
    return bot

def get_response(bot, query, chat_history):
    # Convert chat_history to a list of tuples
    chat_history = [tuple(chat) for chat in chat_history]
    response = bot({"question": query, "chat_history": chat_history})
    return response

def restart():

    print("Chat history and bot knowledge has been cleared!")

    return [], "", gr.State(), "Bot has no knowledge yet! Please enter an initial query to educate the bot."

def main(query, openAI_key, reddit_client_id, reddit_client_secret, reddit_user_agent, chat_history, bot, kb):

    set_openai_key(openAI_key)

    if chat_history == []:

        reddit = set_reddit_keys(reddit_client_id, reddit_client_secret, reddit_user_agent)

        print("Bot knowledge has not been initialised yet! Generating topics...")
        topics = generate_topics(query)
        kb = "Bot now has knowledge of the following topics: [" + "".join([f"{i+1}. {topic} " for i, topic in enumerate(topics)]) + "]"

        print("Fetching relevant comments...")
        comments = get_relevant_comments(reddit, topics)

        print("Embedding relevant comments...")
        retriever = construct_retriever(comments)

        print("Educating bot...")
        bot = construct_bot(retriever)

        print("Bot has been constructed and is ready to use!")

    response = get_response(bot, query, chat_history)

    answer, source_documents = response["answer"], response["source_documents"]

    source_urls = "### Sources\n\nThe following contain sources the bot might have used to answer your last query:\n\n" + "\n\n".join([f'[{x.metadata["title"]} (r/{x.metadata["subreddit"]})](https://www.reddit.com/r/{x.metadata["subreddit"]}/comments/{x.metadata["source"]}/comment/{x.metadata["comment_id"]})' for x in source_documents])

    chat_history.append((query, answer))

    return "", kb, chat_history, source_urls, bot

# Testing only!

title = "Ask Reddit GPT 📜"


with gr.Blocks() as demo:
                
        with gr.Group():
            gr.Markdown(f'<center><h1>{title}</h1></center>')
            gr.Markdown(f"Ask Reddit GPT allow you to ask about and chat with information found on Reddit. The tool uses the Reddit API to build a database of knowledge (stored in a Chroma database) and LangChain to query it. For each response, a list of potential sources are sent back. The first query you sent will take a while as it will need to build a knowledge base based on the topics concerning such query. Subsequent queries on the same topic will be much faster. If however, you would like to ask a question concerning other topics, you will need to clear out the knowledge base. To do this, click the 'Restart knowledge base' button below.")

            with gr.Accordion("Instructions", open=False):
                gr.Markdown('''1. You will need an **Open AI** API key! Get one [here](https://platform.openai.com/account/api-keys).

                2. You will also need **Reddit** credentials! Steps to obtain them:
                * Log in to Reddit.
                * Go [here](https://www.reddit.com/prefs/apps). 
                * Scroll to the bottom.
                * Click "create another app...".
                * Fill in the details as you wish, but make sure you select "script" as the type.
                * Click "create app".
                * Copy the client ID, client secret, and user agent name and paste them in the boxes below.  
                * All done!
                ''')  
        
        with gr.Group():

            with gr.Accordion("Credentials", open=True):
                openAI_key=gr.Textbox(label='Enter your OpenAI API key here:')
                reddit_client_id=gr.Textbox(label='Enter your Reddit client ID here:')
                reddit_client_secret=gr.Textbox(label='Enter your Reddit client secret here:')
                reddit_user_agent=gr.Textbox(label='Enter your Reddit user agent here:')

        with gr.Group():

            kb = gr.Markdown("Bot has no knowledge yet! Please enter an initial query to educate the bot.")
            chat_history = gr.Chatbot()
            bot = gr.State()

            query = gr.Textbox()
            submit = gr.Button("Submit")
            submit.style(full_width=True)

            clear = gr.Button("Restart knowledge base")
            clear.style(full_width=True)

            sources = gr.Markdown()

            submit.click(main, [query, openAI_key, reddit_client_id, reddit_client_secret, reddit_user_agent, chat_history, bot, kb], [query, kb, chat_history, sources, bot])
            clear.click(restart, None, [chat_history, sources, bot, kb], queue=False)

demo.launch()