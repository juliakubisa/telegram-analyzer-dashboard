# connect to te main file
from app import app

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

# load data
data = pd.read_json("/Users/julkakubisa/PycharmProjects/telegram-analysis/data/mama_result.json")
data = pd.json_normalize(data['messages'])

# leave only important columns
data = data.filter(['text','from', 'media_type', 'date'], axis=1)

# convert date to datetime
data["datetime"] = pd.to_datetime(data['date'])

# index messages by datetime
data.index = data['datetime']
data = data.drop(data.loc[:, ['date']], axis=1)

# fill necessary NA
data['media_type'] = data['media_type'].fillna('text')
data = data.dropna()

# extract emojis from messages
def extract_emojis(row):
    message = row.text
    if message is None or type(message) != str:
        return None
    return ''.join(c for c in message if c in emoji.UNICODE_EMOJI['en'])

data["emojis"] = data[["text"]].apply(extract_emojis, axis=1)
total_emojis = list(filter(None, data['emojis']))
emojis_count = Counter("".join(total_emojis)).most_common(5)
emoji_df = pd.DataFrame(emojis_count, columns=['emoji', 'count'])

#figure
fig_emoji = px.pie(emoji_df, hole=.4, values='count', names='emoji',
                   color_discrete_sequence=px.colors.sequential.Magenta)
fig_emoji.update_traces(textposition='inside', textinfo='percent+label')
fig_emoji.update_layout(font=dict(size=15))

# prepare text
data["text"] = data["text"].str.lower()
data["text"] = data["text"].str.replace('[!?.:;,"()-+]', " ")
data['text'] = (data['text'].astype("str")
                .str.replace("ł", "l")
                .str.replace("Ł", "L")
                .str.normalize('NFKD')
                .str.encode('ascii', errors='ignore')
                .str.decode('utf-8'))

# what type of media was sent the most
datatype = data[['media_type', 'datetime']].groupby(['media_type']).count().sort_values(['datetime'], ascending=False)
datatype = datatype.reset_index()
datatype = datatype.rename(columns={'media_type': 'media type', 'datetime': 'no. messsages'})


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
    # print(name, 'sent ', int(words_per_message), ' words, average ', round(words_per_message / user_data.shape[0], 2),
    #    ' words per message')

# number of messages per day
date_df = data.resample("D").apply({'text': 'count'})
date_df.reset_index(inplace=True)

# most active day
day = date_df.max()
exact_day = day[0].strftime("%d/%m/%Y")

# messages per month
date_df_m = data.resample("M").apply({'text': 'count'})
date_df_m.reset_index(inplace=True)

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

# TODO layout
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

# TODO: show wordcloud

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
    name="Unigrams",
    marker={'color': '#dd5182'},
    y=all_grams_df[1].str[0].astype('str'),
    x=all_grams_df[1].str[1],
    orientation='h'),
    row=1, col=1)

fig_ngrams.add_trace(go.Bar(
    name="Bigrams",
    marker={'color': '#e9a9bd'},
    y=all_grams_df[2].str[0].astype('str'),
    x=all_grams_df[2].str[1],
    orientation='h'),
    row=2, col=1)

fig_ngrams.update_layout(
    height=750,
    xaxis_tickangle=45,
    font_size=10,
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig_ngrams.for_each_xaxis(lambda x: x.update(showgrid=False))
fig_ngrams.for_each_yaxis(lambda x: x.update(showgrid=False))
fig_ngrams.update_traces(textfont_size=8)
# TODO

# LAYOUT

# header
header = dbc.Row([
    html.H1(children="Telegram Analyzer", className="header-title"),
    html.Div([
        dcc.DatePickerRange(
            id="date-range",
            min_date_allowed=data.datetime.min().date(),
            max_date_allowed=data.datetime.max().date(),
            start_date=data.datetime.min().date(),
            end_date=data.datetime.max().date(),
            minimum_nights=30,  # TODO monthly distribution error
            display_format='DD/MM/YYYY',
        ),
    ],
        className="date-picker"
    ),
],
    className="header"
)

# statistics
statistics = dbc.Row(
    [
        dbc.Col([
            html.Div(children="Tot. messages", className="stats-title"),
            html.P(className="stats-description", id="tot_mess"),
        ]),
        dbc.Col([
            html.Div(children="Avg. messages/day", className="stats-title"),
            html.P(className="stats-description", id="avg_mess"),
        ]),
        dbc.Col([
            html.Div(children="Avg. words", className="stats-title"),
            html.P(className="stats-description", id="avg_words"),
        ]),
        dbc.Col([
            html.Div(children="Most active day", className="stats-title"),
            html.P(className="stats-description", id="exact_day"),
        ]),
    ],
    className="stats"
)

# timeline
timeline = dbc.Row([
    html.Div([
        html.I(className="bi bi-star-fill"), " Chat timeline",
    ],
        className="graph-title-center"
    ),
    dcc.Graph(id="timeline_graph")
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
                dcc.Graph(id="mess_distr")
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
                # TODO pick the number of emojis shown
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
                dcc.Graph(id="monthly_distr")
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
                html.Div([
                    html.I(className="bi bi-star-fill"), " Messages per month",
                ],
                    className="graph-title"
                ),
                dcc.Graph(figure=fig_ngrams)
            ],
                className='ngrams-container'
            ),
        ],
            md=2
        ),
    ],
    justify="center",
)


# stats
@app.callback(
    [Output("tot_mess", "children"), Output("avg_mess", "children"), Output("exact_day", "children"),
     Output("mess_distr", "figure")],
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)

def update_stats(start_date, end_date):
    mask = (
            (data.datetime >= start_date)
            & (data.datetime <= end_date)
    )
    filtered_data = data.loc[mask, :]
    date_df = filtered_data.resample("D").apply({'text': 'count'}) #messages counted by days
    date_df.reset_index(inplace=True)

    who_sent = filtered_data[['text', 'from']].groupby(['from']).count().sort_values(['text'], ascending=False) #who sent how many
    who_sent = who_sent.reset_index()

    #statistics
    tot_mess = filtered_data['text'].count() #total messages
    sum_days = len(date_df.index) #total number of days
    avg_mess = round(tot_mess / sum_days, 2) #average messages per day
    day = date_df.max()
    exact_day = day[0].strftime("%d/%m/%Y")

    #who sent chart
    mess_distr = px.pie(who_sent, values='text', names='from',
                  color_discrete_sequence=px.colors.sequential.Magenta, labels={'text': 'no. of texts'},
                  )

    mess_distr.update_layout(
        showlegend=False,
        font=dict(size=15)
    )

    mess_distr.update_traces(textposition='inside', textinfo='percent+label')

    return tot_mess, avg_mess, exact_day, mess_distr


# timeline
@app.callback(
    Output("timeline_graph", "figure"),
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def update_timeline(start_date, end_date):
    mask = (
            (date_df.datetime >= start_date)
            & (date_df.datetime <= end_date)
    )
    filtered_data = date_df.loc[mask, :]
    timeline_graph = px.line(filtered_data, x="datetime", y="text",
                             color_discrete_sequence=px.colors.sequential.Magenta,
                             labels={"datetime": "date", "text": "messages"},
                             template="plotly_white")

    timeline_graph.update_layout(title={'x': 0.5},
                                 margin_b=60,
                                 margin_r=30, )

    return timeline_graph


# monthly distribution
@app.callback(
    Output("monthly_distr", "figure"),
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def update_monthly(start_date, end_date):
    mask = (
            (date_df_m.datetime >= start_date)
            & (date_df_m.datetime <= end_date)
    )

    filtered_data_m = date_df_m.loc[mask, :]
    monthly_distr = px.area(filtered_data_m, x="datetime", y="text",
                            color_discrete_sequence=px.colors.sequential.Magenta,
                            labels={"datetime": "date", "text": "messages"},
                            template="plotly_white")

    monthly_distr.update_layout(title={'x': 0.5},
                                margin_b=60,
                                margin_r=30, )
    return monthly_distr


# TODO upload files

# layout
app.layout = dbc.Container(fluid=True, children=[header, statistics, timeline, body],
                           style={'background-color': '#FFFFFF'})

# run the app
if __name__ == '__main__':
    app.run_server(debug=True)
