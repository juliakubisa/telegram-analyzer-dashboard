import dash
from dash import dcc
from dash import html
import pandas as pd
from pandas.io.json import json_normalize
import plotly.express as px
import emoji
import unicodedata
import datetime as dt

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

#total number of messages
sum_msg = data['text'].count()

#who sent how many messages
who_sent = data[['text','from']].groupby(['from']).count().sort_values(['text'], ascending=False)
who_sent = who_sent.reset_index()

# initialize the app
app = dash.Dash(__name__)

# layout
app.layout = html.Div(
    children=[
        html.H1(children="Telegram analyzer", ),
        html.P(
            children="Chat statistics"
                     " between 2011 and 2020 "),
        html.P(children=sum_msg)])

# run the app
if __name__ == '__main__':
    app.run_server(debug=True)
