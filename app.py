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

# extract emojis from messages
def extract_emojis(row):
    message = row.text
    if message is None or type(message) != str:
        return None
    return ''.join(c for c in message if c in emoji.is_emoji['en'])

data["emojis"] = data[["text"]].apply(extract_emojis, axis=1)
print(data)

# initialize the app
app = dash.Dash(__name__)

# app layout
# app.layout = html.Div(
#   children=[
#      html.H1(children="Telegram analyzer",),
#     html.P(
#        children="Chat statistics"
#       " between 2011 and 2020 ")])

# run the app
if __name__ == '__main__':
    app.run_server(debug=True)
