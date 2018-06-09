
# Observations

1. Most tweets from the different media sources are neutral.
2. The NY times overall tends to tweet more negatively.
3. BBC tends to tweet more positively.
4. CNN overall is mostly neutral.


```python
# Dependencies
import tweepy
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target Accounts BBC, CBS, CNN, Fox, and New York times.
target_users = ['@BBC', '@CBSNEWS', '@CNN', '@FOXNEWS', '@NYTIMES']


# Variables for holding sentiments
sentiments = []


for user in target_users:
# Loop through 5 pages of tweets (total 100 tweets)
    oldest_tweet = None
    counter = 1
    for x in range(5):

        # Get all tweets from home feed
        public_tweets = api.user_timeline(user, max_id = oldest_tweet)

        # Loop through all tweets 
        for tweet in public_tweets:
           
            # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])
            compound = results["compound"]
            pos = results["pos"]
            neu = results["neu"]
            neg = results["neg"]
            tweets_ago = counter

            # Get Tweet ID, subtract 1, and assign to oldest_tweet
            oldest_tweet = tweet['id'] - 1

            # Add sentiments for each tweet into a list
            sentiments.append({"Date": tweet["created_at"],
                               "News Outlet" : user,
                               "Compound": compound,
                               "Positive": pos,
                               "Negative": neu,
                               "Neutral": neg,
                               "Tweets Ago": counter,
                              })

            # Add to counter 
            counter += 1
```


```python
#create sentiment datframe
sentiments_pd = pd.DataFrame.from_dict(sentiments)
sentiments_pd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>News Outlet</th>
      <th>Positive</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.7783</td>
      <td>Sat Jun 09 15:02:02 +0000 2018</td>
      <td>0.673</td>
      <td>0.327</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.7717</td>
      <td>Sat Jun 09 14:04:02 +0000 2018</td>
      <td>0.472</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0.528</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>Sat Jun 09 13:24:03 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>Sat Jun 09 13:03:06 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>Sat Jun 09 12:23:25 +0000 2018</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create dataframes for each media source
bbc_df = sentiments_pd.loc[(sentiments_pd["News Outlet"] == "@BBC")]
cbs_df =  sentiments_pd.loc[(sentiments_pd["News Outlet"] == "@CBSNEWS")]
cnn_df =  sentiments_pd.loc[(sentiments_pd["News Outlet"] == "@CNN")]
fox_df = sentiments_pd.loc[(sentiments_pd["News Outlet"] == "@FOXNEWS")]
nytimes_df =  sentiments_pd.loc[(sentiments_pd["News Outlet"] == "@NYTIMES")]
```


```python
# set x and y axis to plot each media source. This will allow colors and labels to be set specifically for each media source
x_bbc = bbc_df["Tweets Ago"]
y_bbc = bbc_df["Compound"]

x_cbs = cbs_df["Tweets Ago"]
y_cbs = cbs_df["Compound"]

x_cnn = cnn_df["Tweets Ago"]
y_cnn = cnn_df["Compound"]

x_fox = fox_df["Tweets Ago"]
y_fox = fox_df["Compound"]

x_nytimes = nytimes_df["Tweets Ago"]
y_nytimes = nytimes_df["Compound"]

plt.scatter(x_bbc, y_bbc, c='lightblue', s=100, label= 'BBC',  marker="o", alpha=0.8, edgecolor = "black", linewidths=1)
plt.scatter(x_cbs, y_cbs, c='green', s=100, label= 'CBSnews', marker="o", alpha=0.8, edgecolor = "black", linewidths=1)
plt.scatter(x_cnn, y_cnn, c='red', s=100, label= 'CNN', marker="o", alpha=0.8, edgecolor = "black", linewidths=1)
plt.scatter(x_fox, y_fox, c='blue', s=100, label= 'FOXnews', marker="o", alpha=0.8, edgecolor = "black", linewidths=1)
plt.scatter(x_nytimes, y_nytimes, s=100, c='yellow', label= 'NYTimes', marker="o", alpha=0.8, edgecolor = "black", linewidths=1)

# # Incorporate the other graph properties
now = datetime.now()
now = now.strftime("%m/%d/%Y")
plt.title(f"Sentiment Analysis of Media Tweets ({now})")
plt.xlim([0,100])
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
# format the legend
plt.legend(bbox_to_anchor=(1.3,.5), loc="lower right", title="Media Sources")
plt.savefig("Sentiment_Analysis_of_Media_Tweets")
plt.show()
                               
```


![png](output_6_0.png)



```python
# Split up our data into groups based upon 'gender'
media_df = sentiments_pd.groupby('News Outlet')

# Find out how many of each gender took bike trips
media_sentiment = media_df['Compound'].mean()

```


```python
now = datetime.now()
now = now.strftime("%m/%d/%Y")
media_chart = media_sentiment.plot(kind="bar",)
media_chart.set_xlabel("Media Source")
media_chart.set_ylabel("Tweet Polarity")
plt.title(f"Overall Media Sentiment Based on Twitter ({now})")
# Save an image of the chart and print it to the screen
plt.savefig("Overall_Media_Sentiment.png")
plt.show()
```


![png](output_8_0.png)

