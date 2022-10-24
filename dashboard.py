# import libraries
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash.dependencies as dd
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import ngrams
from collections import Counter
import warnings
import emoji

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

# connect to te main file
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
data = data.dropna()

#extract emojis from messages
def extract_emojis(row):
    message = row.text
    if message is None or type(message) !=str:
        return None
    return ''.join(c for c in message if c in emoji.UNICODE_EMOJI['en'])

data["emojis"] = data[["text"]].apply(extract_emojis, axis=1)
total_emojis = list(filter(None, data['emojis']))
emojis_count = Counter("".join(total_emojis)).most_common(5)
emoji_df = pd.DataFrame(emojis_count, columns=['emoji', 'count'])


fig_emoji = px.pie(emoji_df, hole=.4, values='count', names='emoji', color_discrete_sequence=px.colors.sequential.Magenta)
fig_emoji.update_traces(textposition='inside', textinfo='percent+label')
fig_emoji.update_layout(font = dict(size = 15))


# prepare text
data["text"] = data["text"].str.lower()
data["text"] = data["text"].str.replace('[!?.:;,"()-+]', " ")
data['text'] = (data['text'].astype("str")
                .str.replace("ł", "l")
                .str.replace("Ł", "L")
                .str.normalize('NFKD')
                .str.encode('ascii', errors='ignore')
                .str.decode('utf-8'))

# total number of messages
sum_msg = data['text'].count()

# messages distribution
#TODO layout
who_sent = data[['text', 'from']].groupby(['from']).count().sort_values(['text'], ascending=False)
who_sent = who_sent.reset_index()

fig1 = px.pie(who_sent, values='text', names='from',
              color_discrete_sequence=px.colors.sequential.Magenta, labels={'text': 'no. of texts'},
              )

fig1.update_layout(
    showlegend=False,
    font = dict(size = 15)
)

fig1.update_traces(textposition='inside', textinfo='percent+label')


# what type of media was sent the most
datatype = data[['media_type', 'datetime']].groupby(['media_type']).count().sort_values(['datetime'], ascending=False)
datatype = datatype.reset_index()
datatype = datatype.rename(columns={'media_type': 'media type', 'datetime': 'no. messsages'})

#datatype.columns('media type', 'no. messages')
#TODO find a way to show it in the dashboard

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

# chat timeline
date_df = data.resample("D").apply({'text': 'count'})
date_df.reset_index(inplace=True)

fig2 = px.line(date_df, x="datetime", y="text",
               color_discrete_sequence=px.colors.sequential.Magenta,
               labels={"datetime": "date", "text": "messages"},
               template="plotly_white")

fig2.update_layout(title={'x': 0.5},
                   margin_b=60,
                   margin_r=30, )

# average messages per day
sum_days = len(date_df.index)
avg_mess = round(sum_msg / sum_days, 2)

# most active day
day = date_df.max()
exact_day = day[0].strftime("%d/%m/%Y")

# messages per month
date_df_m = data.resample("M").apply({'text': 'count'})
date_df_m.reset_index(inplace=True)

fig3 = px.area(date_df_m, x="datetime", y="text",
               color_discrete_sequence=px.colors.sequential.Magenta, labels={"datetime": "date", "text": "messages"},
               template="plotly_white")

fig3.update_layout(title={'x': 0.5},
                   margin_b=60,
                   margin_r=30, )

# messages per hour
data['hour'] = pd.to_datetime(data['datetime'], format='%H:%M').dt.hour

hourly_distr = data[['text', 'hour']].groupby(['hour']).count().sort_values(['hour'], ascending=True)
hourly_distr = hourly_distr.reset_index()

fig4 = px.bar(hourly_distr, x='hour', y='text',
              labels={'text': 'messages'}, template="plotly_white", color='text',
              color_continuous_scale=px.colors.sequential.Magenta)

fig4.update_layout(title={'x': 0.5},
                   margin=dict(b=60, r=30, t=80))

# weekday distribution
data['day_week_num'] = pd.to_datetime(data['datetime'], format='%H:%M').dt.dayofweek
week = data[['text', 'day_week_num']].groupby(['day_week_num']).count()
week = week.reset_index()


def week_name(i):
    l = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return l[i]

week['day_week_num'] = week['day_week_num'].apply(week_name)

#TODO layout
fig5 = px.bar_polar(week, r="text", theta="day_week_num",
                    color="text", template="plotly_white", labels={'text': 'messages'},
                    color_continuous_scale=px.colors.sequential.Magenta)

fig5.update_layout(title={'x': 0.5},
                   margin=dict(b=60, r=30, t=80), polar=dict(radialaxis_showticklabels=False))

# text mining
polish_stopwords = open("/Users/julkakubisa/PycharmProjects/telegram-analysis/data/polish.stopwords.txt",
                        "r").read().replace("\n'", " ").split()

# new dataframe for text mining
df2 = data[['text', 'from']]


# tokenize
def tokenize(text):
    text_tokens = word_tokenize(text)
    tokens_processed = [j for j in text_tokens if j not in polish_stopwords and len(j) > 1]
    text = " ".join(tokens_processed)
    return text


df2['tokenized'] = df2['text'].apply(tokenize)


# split words
splitted = " ".join(df2['tokenized']).split()

# wordcloud
# cloud = WordCloud(background_color='white', max_font_size=150,
#                   ).generate(df2['tokenized'].to_string())

#TODO: show wordcloud

# ngrams
def ngrams_df(num_grams, splitted):
    all_grams = pd.DataFrame()

    for i in range(1, num_grams + 1):
        n_grams = ngrams(splitted, i)
        ngrams_count = Counter(n_grams).most_common(10)
        all_grams[i] = ngrams_count

    return all_grams

all_grams_df = ngrams_df(2, splitted)

fig_ngrams = make_subplots(rows=2, cols=1)

fig_ngrams.add_trace(go.Bar(
    name = "Unigrams",
    y=all_grams_df[1].str[0].astype('str'),
    x=all_grams_df[1].str[1],
    orientation='h'),
    row=1, col=1)

fig_ngrams.add_trace(go.Bar(
    name = "Bigrams",
    y=all_grams_df[2].str[0].astype('str'),
    x=all_grams_df[2].str[1],
    orientation='h'),
    row=2, col=1)

fig_ngrams.update_layout(height=750, xaxis_tickangle=45, title_text="Frequently used words", font_size=10, template = "plotly_white", barmode='stack')
fig_ngrams.update_traces(textfont_size=8)

fig_ngrams.show()

#TODO list of lists shown in a table

# LAYOUT

# header
header = dbc.Row([
    html.H1(children="Telegram Analyzer", className="header-title"),
    html.P(children="Date range x - x", className="header-description"), #TODO actual date range
],
    className="header"
)

# statistics
statistics = dbc.Row(
    [
        dbc.Col([
            html.Div(children="Tot. messages", className="stats-title"),
            html.P(children=sum_msg, className="stats-description"),
        ]),
        dbc.Col([
            html.Div(children="Avg. messages/day", className="stats-title"),
            html.P(children=avg_mess, className="stats-description"),
        ]),
        dbc.Col([
            html.Div(children="Avg. words", className="stats-title"),
            html.P(children="3.52", className="stats-description"),
        ]),
        dbc.Col([
            html.Div(children="Most active day", className="stats-title"),
            html.P(children=exact_day, className="stats-description"),
        ]),
    ],
    className="stats"
)

#timeline
timeline = dbc.Row([
    html.Div([
        html.I(className="bi bi-star-fill"), " Chat timeline",
    ],
        className="graph-title-center"
    ),
    dcc.Graph(figure=fig2)
],
)


# body
body = dbc.Row(
    [
        dbc.Col([
            dbc.Row([
                html.Div([
                    html.I(className="bi bi-star-fill"), "  Messages distribution",
                ],
                    className="graph-title"
                ),
                dcc.Graph(figure=fig1)
            ],
            ),
            dbc.Row([
                html.Div([
                    html.I(className="bi bi-star-fill"), " Weekday distribution",
                ],
                    className="graph-title"
                ),
                dcc.Graph(figure=fig5)
            ],
            ),
            dbc.Row([
                html.Div([
                    html.I(className="bi bi-star-fill"), " Emoji distribution",
                ],
                    className="graph-title"
                ),
                dcc.Graph(figure=fig_emoji)
                #TODO pick the number of emojis shown
            ],
            ),
        ],
            md=3,
            className="pie-container"),

        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="bi bi-star-fill"), "  Messages per hour",
                ],
                    className="graph-title"
                ),
                dcc.Graph(figure=fig4)
            ],
                className="right-container"
            ),
            html.Div([
                html.Div([
                    html.I(className="bi bi-star-fill"), " Messages per month",
                ],
                    className="graph-title"
                ),
                dcc.Graph(figure=fig3)
            ],
                className="right-container"
            ),

        ],
            md=5,
        ),
        dbc.Col([
            html.Div([
            dbc.Table.from_dataframe(datatype, hover=True, className="table-style", index=False)
                ],
                className='tables-container'
            ),
            html.Div([
                dcc.Graph(figure=fig_ngrams)
                ])
        ],
            md=2
       ),
    ],
    justify="center",
)


# layout
app.layout = dbc.Container(fluid=True, children=[header, statistics, timeline, body],
                           style={'background-color': '#FFFFFF'})


#callbacks



# run the app
if __name__ == '__main__':
    app.run_server(debug=True)
