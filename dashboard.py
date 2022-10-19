# import libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import re
import numpy as np
from nltk.tokenize import word_tokenize

#connect to te main file
from app import app

# load data
data = pd.read_json("/Users/julkakubisa/PycharmProjects/telegram-analysis/data/mama_result.json")
data = pd.json_normalize(data['messages'])

# drop columns
data = data.drop(data.loc[:, ['type', 'actor', 'actor_id', 'action', 'from_id', 'photo', 'width',
                              'height', 'reply_to_message_id', 'file', 'thumbnail', 'mime_type', 'duration_seconds',
                              'forwarded_from', 'discard_reason',
                              'emoticon', 'contact_vcard', 'contact_information.first_name',
                              'contact_information.last_name', 'contact_information.phone_number',
                              'edited', 'via_bot']], axis=1)
data = data.drop(data.loc[:, ['id', 'sticker_emoji']], axis=1)

# convert date to datetime
data["datetime"] = pd.to_datetime(data['date'])

# index messages by datetime
data.index = data['datetime']
data = data.drop(data.loc[:, ['date']], axis=1)

# fill necessary NA
data['media_type'] = data['media_type'].fillna('text')

# cleaning text
data["text"] = data["text"].str.lower()

# replace special characters with blank spaces
data["text"] = data["text"].str.replace('[!?.:;,"()-+]', " ")

# normalize polish characters
data['text'] = (data['text'].astype("str")
                .str.replace("ł", "l")
                .str.replace("Ł", "L")
                .str.normalize('NFKD')
                .str.encode('ascii', errors='ignore')
                .str.decode('utf-8'))

# total number of messages
sum_msg = data['text'].count()

# messages distribution
who_sent = data[['text', 'from']].groupby(['from']).count().sort_values(['text'], ascending=False)
who_sent = who_sent.reset_index()

fig1 = px.pie(who_sent, values='text', names='from', title="% of sent messages",
              color_discrete_sequence=px.colors.sequential.Magenta, labels={'text': 'no. of texts'},
              )

fig1.update_layout(title={'x': 0.5},
                   legend=dict(
                       yanchor="top",
                       y=1.1,
                       x=0.45,
                       orientation="h"),
                   margin=dict(b=10)
                   )

# what type of media was sent the most
datatype = data[['media_type', 'datetime']].groupby(['media_type']).count().sort_values(['datetime'], ascending=False)


# get the number of words in messages
def word_count(row):
    message = row.text
    if message is None or type(message) != str:
        return None
    return re.sub("[^\w]", " ", message).split().__len__()


# create a dataframe with word count
data['word_count'] = data[['text']].apply(word_count, axis=1)
data_wc = data[['from', 'word_count', 'datetime']].copy()

data = data.drop(['word_count'], axis=1)

# number of words per message
people = data_wc['from'].unique()
for name in people:
    user_data = data_wc[data_wc["from"] == name]
    words_per_message = np.sum(user_data['word_count'])
    print(name, 'sent ', int(words_per_message), ' words, average ', round(words_per_message / user_data.shape[0], 2),
          ' words per message')

# messages over days
date_df = data.resample("D").apply({'text': 'count'})
date_df.reset_index(inplace=True)

fig2 = px.line(date_df, x="datetime", y="text", title="Chat timeline",
               color_discrete_sequence=px.colors.sequential.Magenta,
               labels={"datetime": "date", "text": "messages"},
               template="plotly_white")

fig2.update_layout(title={'x': 0.5},
                   margin_b=60,
                   margin_r=30, )

# average messages per day
sum_days = len(date_df.index)
avg_mess = round(sum_msg / sum_days, 2)

# day with the most messages
day = date_df.max()
exact_day = day[0].strftime("%d/%m/%Y")

# messages over month
date_df_m = data.resample("M").apply({'text': 'count'})
date_df_m.reset_index(inplace=True)

fig3 = px.area(date_df_m, x="datetime", y="text", title="Number of messages per month",
               color_discrete_sequence=px.colors.sequential.Magenta, labels={"datetime": "date", "text": "messages"},
               template="plotly_white")

fig3.update_layout(title={'x': 0.5},
                   margin_b=60,
                   margin_r=30, )

# busiest hour
data['hour'] = pd.to_datetime(data['datetime'], format='%H:%M').dt.hour

hourly_distr = data[['text', 'hour']].groupby(['hour']).count().sort_values(['hour'], ascending=True)
hourly_distr = hourly_distr.reset_index()

fig4 = px.bar(hourly_distr, x='hour', y='text', title="Number of messages per hour",
              labels={'text': 'messages'}, template="plotly_white", color='text',
              color_continuous_scale=px.colors.sequential.Magenta)

fig4.update_layout(title={'x': 0.5},
                   margin=dict(b=60, r=30, t=80))

# busiest day of the week
data['day_week_num'] = pd.to_datetime(data['datetime'], format='%H:%M').dt.dayofweek
week = data[['text', 'day_week_num']].groupby(['day_week_num']).count()
week = week.reset_index()

def week_name(i):
    l = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return l[i]


week['day_week_num'] = week['day_week_num'].apply(week_name)

fig5 = px.bar_polar(week, r="text", theta="day_week_num", title="Number of messages per weekday",
                    color="text", template="plotly_white", labels={'text': 'messages'},
                    color_continuous_scale=px.colors.sequential.Magenta)

fig5.update_layout(title={'x': 0.5},
                   margin=dict(b=60, r=30, t=80), polar=dict(radialaxis_showticklabels=False))

# text mining
polish_stopwords = open("/Users/julkakubisa/PycharmProjects/telegram-analysis/data/polish.stopwords.txt",
                        "r").read().replace("\n'", " ").split()

# new dataframe for text mining
df2 = data[['text', 'from']]
df2 = df2.reset_index()

def tokenize(text):
    text_tokens = word_tokenize(text)
    tokens_processed = [j for j in text_tokens if j not in polish_stopwords and len(j) > 1]
    text = " ".join(tokens_processed)
    return text

df2['tokenized'] = df2['text'].apply(tokenize)

# split words
splitted = " ".join(df2['tokenized']).split()

# wordcloud
cloud = WordCloud(background_color='white', max_font_size=150,
                  ).generate(df2['tokenized'].to_string())

# layout
app.layout = html.Div(
    children=[
        html.H1(
            children="Telegram analyzer",
            className= "header-title"),
        html.P(
            children="Chat statistics between 2011 and 2022",
            className= "header-description"),
    ],
    className="header"
)

# run the app
if __name__ == '__main__':
    app.run_server(debug=True)
