import pickletools
import re
import configparser
from textblob import TextBlob
import tweepy
import pandas as pd
import time
import sys
import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')

keyword= 'Valor inicial'


# Create a Function to keys defintion
def keys_definition():
    """returns all the keys from twitter from a config.ini file and a configparser package 

    Returns:
        tuple: All the keys for twitter
    """
    config= configparser.ConfigParser(interpolation=None)
    config.read('config.ini')
    api_key = config['twitter']['api_key']
    api_key_secret = config['twitter']['api_key_secret']
    access_token = config['twitter']['access_token']
    access_token_secret = config['twitter']['access_token_secret']
    bearer_token = config['twitter']['bearer_token']
    return api_key, api_key_secret, access_token, access_token_secret, bearer_token


def get_tweets(hashtag, language='en', num_tweets=1000):
    """Get tweets from a tweepy.paginator (flatten)

    Args:
        hashtag (bool): A hashtag to search for
        language (str, optional): The language. Defaults to 'en'
        num_tweets (int, optional): Number of tweet you want to search for
    """
    global keyword
    keyword= hashtag
    query = f'{keyword} -is:retweet lang:'+language
    #connecting to the twitter API using clent and the bearer_token credentials from config.py
    
    client = tweepy.Client(bearer_token=keys_definition()[4])

    #using tweepy paginator to get over 100 last tweets from twitter api
    tweets = []
    for tweet in tweepy.Paginator(client.search_recent_tweets,
                                    query = query,                             
                                    tweet_fields = ['id','created_at', 'public_metrics', 'text', 'source'],
                                    max_results = 100).flatten(limit=num_tweets):
        tweets.append(tweet)
    return tweets
    
    
    
    
# Create a Function to clean the tweets
def CleanTweets(text, keyword):
    """Clean Tweets from hashtags, mentions, and hyperlinks and the word wich we look for  for sentyment analysis

    Args:
        text (str): the text of the tweet
        keyword (srt): the hastag of the tweet

    Returns:
        str: The cleaned text
    """
    clean_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-zñáéíóú \t])|(\w+:\/\/\S+)"," ",text).split())
    return clean_text
    
# Create a Function to transform tweets
def tweets_toframe(tweets):
    """Converts tweets from tweepy.paginator in data frame type

    Args:
        tweets (paginator flatten object): tweets from tweepy.paginator

    Returns:
        pd.DataFrame: a data frame of tweets with the columns indicated below
    """
    result=[]
    for tweet in tweets:
            result.append({'id': tweet.id,
                           'text': tweet.text,
                           'clean_tweet' : CleanTweets(tweet.text, keyword),
                           'created_at': tweet.created_at,
                           'source':tweet.source,
                           'retweets': tweet.public_metrics['retweet_count'],
                           'replies': tweet.public_metrics['reply_count'],
                           'likes': tweet.public_metrics['like_count'],
                           'quote_count': tweet.public_metrics['quote_count']
                      })

    df = pd.DataFrame(result)
    return df

def sentimentAnalysis(df):
    """performing the sentiment analysis using a base BERT model. 
    using a transformers model "bert" to perform the sentiment analysis on the clean_tweets column.

    Args:
        df (pd.Dataframe): dataframe of tweets. You can use tweets_toframe() to get it

    Returns:
        Dict: a sentiment analysis of the tweets clean text
    """
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    res = df['clean_tweet'].apply(lambda x: classifier(x[:512]))
    return res

def sentimentToDf(df,res):
    """ function to add the list resulting from the analysis to the original dataframe as score, 
    sentiment and stars. The sentiment is either negative, positive or neutral, and the number 
    of stars go from 1 to 5

    Args:
        df (pd.DataFrame): a dataFrame got from tweets_toframe() function
        res (Dict): list of sentiments for every tweet

    Returns:
        pd.DataFrame: A original dataframe with the added sentiment
    """
    tweets_stars = []
    tweets_scores = []
    tweets_sentiment = []
    #looping over the list of result to unpack it into the original tweets dataframe
    for i in range(res.size):
        tweets_stars.append(int(float(res[i][0]['label'].split()[0])))
        tweets_scores.append(res[i][0]['score'])
        if res[i][0]['label'] == '4 stars' or res[i][0]['label'] == '5 stars':
            tweets_sentiment.append('positive')
        elif res[i][0]['label'] == '1 star' or res[i][0]['label'] == '2 stars':
            tweets_sentiment.append('negative')
        else :
            tweets_sentiment.append('neutral')
    df['scores'] = tweets_scores
    df['sentiment'] = tweets_sentiment  
    df['stars'] = tweets_stars
    return df


def generate_stopwords():
    """Generate stopwords from nltk.corpus in spanish, english and French

    Returns:
        list: the stopwords in the above mentioned languages
    """
    stopword_en = nltk.corpus.stopwords.words('english')
    stopword_es = nltk.corpus.stopwords.words('spanish')
    stopword_fr = nltk.corpus.stopwords.words('french')
    
    return stopword_en + stopword_es + stopword_fr


def createWordCloud(df,clm_name):
    """Creates a Wordcloud from a text column of a dataframe

    Args:
        df (pd.DataFrame): A dataframe with a cleaned text column from tweets Got from sentimentToDf()
        clm_name (pd.Series): Column of cleaned text from the dataframe

    Returns:
        Wordcloud: The wordcloud of all the tweets
    """
     
    text = " ".join(line for line in df[clm_name])
    
    for word in keyword.split(' '):
        text = ' '.join(re.sub(word.capitalize()," ",text).split())
        text = ' '.join(re.sub(word.lower()," ",text).split())
        text = ' '.join(re.sub(word," ",text).split())
        
    # Create the wordcloud object
    wordcloud = WordCloud(stopwords= generate_stopwords(),width=980, height=580, margin=0,collocations = False, background_color = 'white').generate(text)
    # Display the generated image:
    plt.figure(figsize=(12,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()
    return plt 

def showReport(df):   
    """ Creating report function to get insight from the analyzed tweets
    creating a function to show the result of the sentiment analysis from the final df  

    Args:
        df (pd.DataFrame): the data frame cleaned we get from sentimentToDf
    """    
    print(f'* the tweets show that the sentiment around "{keyword}" is mainly {df.groupby(by="sentiment").id.count().sort_values(ascending=False).index[0] }')
    print(f'* this is how the overall sentiment and stars ratings breakdown on the {len(df)} total records we recovered : ')
    print(df.groupby(["stars"]).count()['id'])
    
    # Build the percentage of star count reviews by category pie chart.
    star_perc = 100 * df.groupby(["stars"]).count()['id'] / len(df)
    labels=sorted([str(element)+" stars" for element in df['stars'].unique().tolist()])
    color_labels={'1 stars':'red','2 stars':'orange','3 stars':'gold','4 stars':'turquoise','5 stars':'green'}
    plt.pie(star_perc,labels=labels,
            colors=[color_labels[k] for k in color_labels.keys() if k in labels],
            explode=[0.05, 0.05, 0.05, 0.05, 0.05][0:len(labels)],
            autopct='%1.1f%%',
            shadow=True, startangle=150)
    plt.title("percentage of Total tweets by star ratings")
    plt.show()
    # Show Figure
    # Build the sentiment reviews by category pie chart.
    sent_perc = 100 * df.groupby(["sentiment"]).count()['id'] / len(df) 
    color_sentiment={'negative':'red','neutral':'gold','positive':'green'}
    sentiments=sorted([element for element in df['sentiment'].unique().tolist()])
    plt.pie(sent_perc,
            colors=[color_sentiment[k] for k in color_sentiment.keys() if k in sentiments],
            explode=[0.05, 0.05, 0.05][0:len(sentiments)],
            autopct='%1.1f%%',
            shadow=True, startangle=150)
    plt.title("percentage of total tweets by sentiment ")
    # Show Figure
    plt.show()




# Function to get Subjetivity
def getSubjetivity(text):
    return TextBlob(text).sentiment.subjectivity

# Function to get polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Function to get polarity
def sent_polarity(df,column):
    df['Subjectivity']= df[column].apply(getSubjetivity)
    df['Polarity']= df[column].apply(getPolarity)
    return df
