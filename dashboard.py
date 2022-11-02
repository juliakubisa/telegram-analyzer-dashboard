# Connect to te main file
from app import app

# Libraries for dash front-end
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Libraries for the data analysis
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import ngrams
from collections import Counter
import warnings
import emoji

# Exclude warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data
data = pd.read_json("/Users/julkakubisa/PycharmProjects/telegram-analysis/data/mama_result.json")
data = pd.json_normalize(data['messages'])
data = data.filter(['text','from', 'media_type', 'date'], axis=1)       # Leave only important columns

# Index messages by datetime
data["datetime"] = pd.to_datetime(data['date'])
data.index = data['datetime']
data = data.drop(data.loc[:, ['date']], axis=1)

# Fill necessary NA
data['media_type'] = data['media_type'].fillna('text')
data = data.dropna()

# Extract emojis from messages
def extract_emojis(row):
    message = row.text
    if message is None or type(message) != str:
        return None
    return ''.join(c for c in message if c in emoji.UNICODE_EMOJI['en'])        # Returns emojis that appeared in message

data["emojis"] = data[["text"]].apply(extract_emojis, axis=1)
total_emojis = list(filter(None, data['emojis']))       # Creates a list of each sequence of emojis used
emojis_count = Counter("".join(total_emojis)).most_common(5)        # Counts individual appearances of emoji
emoji_df = pd.DataFrame(emojis_count, columns=['emoji', 'count'])       # Creates a dataframe for emojis

# Plot an emoji figure
fig_emoji = px.pie(emoji_df, hole=.4, values='count', names='emoji',
                   color_discrete_sequence=px.colors.sequential.Magenta)
fig_emoji.update_traces(textposition='inside', textinfo='percent+label')
fig_emoji.update_layout(font=dict(size=15))


# Prepare text
data["text"] = data["text"].str.lower()
data["text"] = data["text"].str.replace('[!?.:;,"()-+]', " ")
data['text'] = (data['text'].astype("str")
                .str.replace("ł", "l")
                .str.replace("Ł", "L")
                .str.normalize('NFKD')
                .str.encode('ascii', errors='ignore')
                .str.decode('utf-8'))

# Gets the number of word per message
def word_count(row):
    message = row.text
    if message is None or type(message) != str:
        return None
    return re.sub("\W", " ", message).split().__len__()     # Splits by whole words that aren't [A-Za-z0-9_]

# Creates a list for naming days of the week
def week_name(i):
    l = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return l[i]

### Text mining
# Import stopwords
polish_stopwords = open("/Users/julkakubisa/PycharmProjects/telegram-analysis/data/polish.stopwords.txt",
                        "r").read().replace("\n'", " ").split()

# Tokenize function
def tokenize(text):
    text_tokens = word_tokenize(text)
    tokens_processed = [j for j in text_tokens if j not in polish_stopwords and len(j) > 1]
    text = " ".join(tokens_processed)
    return text

### DASH LAYOUT
header = dbc.Row([
    html.H1(children="Telegram Analyzer", className="header-title"),
    dcc.Upload([
        'Drag and Drop or ',
        html.A('Select a File')
    ], className="upload"
    ),
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


timeline = dbc.Row([
    html.Div([
        html.I(className="bi bi-star-fill"), " Chat timeline",
    ],
        className="graph-title-center"
    ),
    dcc.Graph(id="timeline_graph")
],
)


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
                dcc.Graph(id="week_distr")
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
                dcc.Graph(id="hour_distr")
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
                html.Div(id="table_cont"),
            ],
                className='tables-container'
            ),
            html.Div([
                html.Div([
                    html.I(className="bi bi-star-fill"), " Messages per month",
                ],
                    className="graph-title"
                ),
                dcc.Graph(id="fig_ngrams")
            ],
                className='ngrams-container'
            ),
        ],
            md=2
        ),
    ],
    justify="center",
)

### CALLBACKS
@app.callback(
    [Output("tot_mess", "children"), Output("avg_mess", "children"), Output("avg_words", "children"),
     Output("exact_day", "children"), Output("mess_distr", "figure"), Output("hour_distr", "figure"),
     Output("monthly_distr", "figure"), Output("timeline_graph", "figure"), Output("week_distr", "figure"),
     Output("fig_ngrams", "figure"), Output("table_cont", "children")],
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)

def update_stats(start_date, end_date):
    mask = (        # Range of dates for creating graphs
            (data.datetime >= start_date)
            & (data.datetime <= end_date)
    )
    filtered_data = data.loc[mask, :]       # Data filtered depending on chosen range

    # Group messages by days
    date_df = filtered_data.resample("D").apply({'text': 'count'})
    date_df.reset_index(inplace=True)

    # Group messages by months
    date_df_m = filtered_data.resample("M").apply({'text': 'count'})        # Messages counted by months
    date_df_m.reset_index(inplace=True)

    # Count messages by each author
    who_sent = filtered_data[['text', 'from']].groupby(['from']).count().sort_values(['text'],
                                                                                     ascending=False).reset_index()

    # Apply number of words for each message
    filtered_data['word_count'] = filtered_data[['text']].apply(word_count, axis=1)

    # Count messages by each hour
    filtered_data['hour'] = pd.to_datetime(filtered_data['datetime'], format='%H:%M').dt.hour
    hourly_distr = filtered_data[['text', 'hour']].groupby(['hour']).count().sort_values(['hour'],
                                                                                         ascending=True).reset_index()

    # Basic statistics
    tot_mess = filtered_data['text'].count()        # Count the total number of messages
    sum_days = len(date_df.index)       # Count the number of appearances by days
    avg_mess = round(tot_mess / sum_days, 2)        # Average messages by day
    avg_words = round(sum(filtered_data['word_count'])/len(data.index), 2)  # Average words per message
    day = date_df.max()
    exact_day = day[0].strftime("%d/%m/%Y")     # Exact day with the most messages

    # Messages distribution chart
    mess_distr = px.pie(who_sent, values='text', names='from',
                  color_discrete_sequence=px.colors.sequential.Magenta, labels={'text': 'no. of texts'},
                  )

    mess_distr.update_layout(
        showlegend=False,
        font=dict(size=15)
    )

    mess_distr.update_traces(textposition='inside', textinfo='percent+label')

    # Hourly distribution chart
    hour_distr = px.bar(hourly_distr, x='hour', y='text',
                  labels={'text': 'messages'}, template="plotly_white", color='text',
                  color_continuous_scale=px.colors.sequential.Magenta)

    hour_distr.update_layout(title={'x': 0.5},
                       margin=dict(b=60, r=30, t=80))

    # Monthly distribution chart
    monthly_distr = px.area(date_df_m, x="datetime", y="text",
                            color_discrete_sequence=px.colors.sequential.Magenta,
                            labels={"datetime": "date", "text": "messages"},
                            template="plotly_white")

    monthly_distr.update_layout(title={'x': 0.5},
                                margin_b=60,
                                margin_r=30, )

    # Timeline chart
    timeline_graph = px.line(date_df, x="datetime", y="text",
                                 color_discrete_sequence=px.colors.sequential.Magenta,
                                 labels={"datetime": "date", "text": "messages"},
                                 template="plotly_white")

    timeline_graph.update_layout(title={'x': 0.5},
                                     margin_b=60,
                                     margin_r=30, )

    # Weekly distribution
    filtered_data['day_week_num'] = pd.to_datetime(filtered_data['datetime'], format='%H:%M').dt.dayofweek  # Extracts the day of the week
    week = filtered_data[['text', 'day_week_num']].groupby(['day_week_num']).count().reset_index()
    week['day_week_num'] = week['day_week_num'].apply(week_name)

    # Weekly distribution chart
    week_distr = px.bar_polar(week, r="text", theta="day_week_num",
                        color="text", template="plotly_white", labels={'text': 'messages'},
                        color_continuous_scale=px.colors.sequential.Magenta)

    week_distr.update_layout(title={'x': 0.5},
                       margin=dict(b=60, r=30, t=80), polar=dict(radialaxis_showticklabels=False))

    # Ngrams chart
    df2 = filtered_data[['text', 'from']]
    df2['tokenized'] = df2['text'].apply(tokenize)

    # Split words
    splitted = " ".join(df2['tokenized']).split()

    def ngrams_df(num_grams, splitted):
        all_grams = pd.DataFrame()

        for i in range(1, num_grams + 1):
            n_grams = ngrams(splitted, i)
            ngrams_count = Counter(n_grams).most_common(10)
            all_grams[i] = ngrams_count

        return all_grams

    all_grams_df = ngrams_df(2, splitted)

    # Ngrams charts
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
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig_ngrams.for_each_xaxis(lambda x: x.update(showgrid=False))
    fig_ngrams.for_each_yaxis(lambda x: x.update(showgrid=False))
    fig_ngrams.update_traces(textfont_size=8)

    #Media type
    datatype = filtered_data[['media_type', 'datetime']].groupby(['media_type']).count().sort_values(['datetime'],
                                                                                                     ascending=False).reset_index()
    datatype = datatype.rename(columns={'media_type': 'media type', 'datetime': 'no. messsages'})
    table = dbc.Table.from_dataframe(datatype, hover=True, className="table-style", index=False)

    return tot_mess, avg_mess, avg_words, exact_day, mess_distr, hour_distr,\
           monthly_distr, timeline_graph, week_distr, fig_ngrams, table

# Create a layout
app.layout = dbc.Container(fluid=True, children=[header, statistics, timeline, body],
                           style={'background-color': '#FFFFFF'}
)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


# TODO upload files
