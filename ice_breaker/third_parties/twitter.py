import os
from dotenv import load_dotenv
import requests

load_dotenv()

def scrape_user_tweets(username, name_tweets=5, mock: bool=True):
  """
  Scrapes a tweeter user's original tweets (i.e.,  not retweets or replies) and returns them as a list of dictionaries.
  Each dictionary has three fields: "time_posted" (relative to now), "text", and "url".
  """

  tweet_list = []

  if mock:
    EDEN_TWEETER_GIST = "https://gist.githubusercontent.com/emarco177/827323bb599553d0f0e662da07b9ff68/raw/57bf38cf8acce0c87e060f9bb51f6ab72098fbd6/eden-marco-twitter.json"
    tweets = requests.get(EDEN_TWEETER_GIST).json()

    for tweet in tweets:
      tweet_dict = {}
      tweet_dict["text"] = tweet["text"]
      tweet_dict["url"] = f"https://twitter.com/{username}/status/{tweet['id']}"
      tweet_list.append(tweet_dict)
    
    return tweet_list

if __name__ == '__main__':
    username = "EdenEmarco177"
    tweets = scrape_user_tweets(username, name_tweets=5, mock=True)
    print(tweets)