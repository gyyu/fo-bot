import sys
import time
import pandas as pd
import re
from textgenrnn import textgenrnn
import markovify
import praw
#import config

with open('input.csv') as input_fd:
    text = input_fd.read()

textgen = textgenrnn('textgenrnn_weights.hdf5')
markov = markovify.Text(text)

def filterResponses(text):
    text = re.sub(r'http\S+', '', text)   # Remove URLs
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # Remove @ mentions
    text = text.strip(" ")   # Remove whitespace resulting from above
    text = re.sub(r' +', ' ', text)   # Remove redundant spaces

    return text

def getMarkovReply():
    print("MC:")
    print(filterResponses(markov.make_sentence()))

def getRNNReply():
    print("RNN:")
    textgen.generate(1,temperature=1.2)

if __name__ == "__main__":
    while True:
        user_input = input(">")
        getMarkovReply()
        getRNNReply()

'''
Reddit Bot Portion
'''

'''
reddit = praw.Reddit(client_id = config.client_id,
                     client_secret = config.client_secret,
                     password = config.password,
                     user_agent = config.user_agent,
                     username = config.username)

def get_subreddits(input_file):
    df = pd.read_csv(input_file)
    subreddits = df['Subreddit']
    return [entry for entry in subreddits if len(entry) > 0]

if __name__ == "__main__":
    input_file = sys.argv[1]
    subreddits = get_subreddits(input_file)
    hate_words = sys.argv[2]
    hate_dict = []
    with open(hate_words, 'rb') as hate_fd:
        hate_dict = reduce(lambda x,y: list(csv.reader(hate_fd)))
        hate_dict = map(lambda x: x.lower(), hate_dict)

    while True:
        for s in subreddits:
            for comment in reddit.subreddit(s).comments(line=1000):
                text = comment.body
                if any(word in text for word in hate_dict):
                    comment.reply(getMarkovReply())
                    comment.reply(getRNNReply())
                    # Keep track of seen IDs to avoid repeated comments
        time.sleep(3600)

'''
