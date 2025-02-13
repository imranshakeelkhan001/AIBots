import pandas as pd
import ast

async def retrive_history(id):
    history = pd.read_csv('chat_history.csv')
    if int(id) in list(history['id']):

        chat_history_str = history.loc[history["id"] == id, "chat_history"].values[0]
        chat_history_str = ast.literal_eval(chat_history_str)
        print(chat_history_str[1:])
        return chat_history_str[1:]

    else:
        return {"error": "No valide user found"}

# retrive_history(15)