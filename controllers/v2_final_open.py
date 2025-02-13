import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI()
import pandas as pd
import os

def store_chat_history(chat_messages, chat_id, csv_file="open_chat_history.csv"):
    """
    Stores OpenAI-style chat messages in a pandas DataFrame against a given ID.
    If the ID exists, the chat will be replaced. Otherwise, a new row will be created.
    The DataFrame is saved to a CSV file.

    Args:
        chat_messages (list): OpenAI-style chat messages (list of dicts).
        chat_id (str or int): Unique ID to associate with the chat.
        csv_file (str): Path to the CSV file where the chat history will be stored. Default is "chat_history.csv".
    """
    # Check if the CSV file exists
    if os.path.exists(csv_file):
        # Load existing chat history
        chat_df = pd.read_csv(csv_file)
    else:
        # Create a new DataFrame if the file does not exist
        chat_df = pd.DataFrame(columns=["id", "chat_history"])

    # Check if the ID exists in the DataFrame
    if chat_id in chat_df["id"].values:
        # Update the existing row
        chat_df.loc[chat_df["id"] == chat_id, "chat_history"] = str(chat_messages)
    else:
        # Add a new row
        new_row = {"id": chat_id, "chat_history": str(chat_messages)}
        chat_df = pd.concat([chat_df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    chat_df.to_csv(csv_file, index=False)
    print(f"Chat history for ID {chat_id} has been saved to {csv_file}.")





system_message =     {
        "role": "system",
        "content": (
            "You are a helpful DESCON HR assistant.Start conversation with greetings by introducting your name only 1 time ' I am Descon HR Assistant'"
            " and your response should be conversational."
            "'do not ask are you looking for a job description for a specific role' or Hello! I am Descon HR Assistant in every response,"
            "you will greet on every response provided by user with connecting words. you will not ask multiple questions from user."
            "Your job is to provide the Job Description.For this purpose, "
            "you will get the details from the user by asking 3 ~4 questions if needed, each question should be engaging and connected, and greet user by receving the information for each question and based on these details "
            "you will use these details for generating tailored and professional job description"
            "try to use professional skills name, certification name, in skill section if needed."
            " before generating job desciption. Do not ask the user multiple questions in one response"
            " and provide a complete job description with professional look by  including professional  job description,key responsibilities, and up to date required skills,qualifications and software. mentioned about certifications if user asked to enter or when it is needed to enter."
            " After providing the job description.User may ask you for some changes, try to make these changes specific only in that job description that you just provided. Note:  'Do not provide new job description' "
            "unless the user want any other category job description that user asked and provide the desired output that user want."
            "As you are a DESCON HR Assistant, of a company DESCON, so include company name of DESCON in job description. as you are creating job description for your comoany."
        )
    }



def job_description_bot(message):
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=message
    )
    return completion




def get_chat_by_id(chat_id, csv_file="open_chat_history.csv"):
    """
    Retrieves the chat corresponding to the given ID from a CSV file.

    Args:
        chat_id (str or int): The unique ID for the chat.
        csv_file (str): Path to the CSV file where the chat history is stored.

    Returns:
        generator: Yields individual messages (dict) from the chat history if the ID is found.
        None: If the ID is not found in the file.
    """
    try:
        # Load the chat history from the CSV file
        history = pd.read_csv(csv_file)

        # Check if the ID exists in the DataFrame
        if chat_id in history["id"].values:
            # Retrieve the corresponding chat history
            chat_history_str = history.loc[history["id"] == chat_id, "chat_history"].values[0]

            # Convert the string representation of the chat history back to a list
            chat_history = eval(chat_history_str)

            # Yield individual messages as separate dictionaries
            for message in chat_history:
                yield message
        else:
            print(f"Chat with ID {chat_id} not found.")
            return None
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None







async def jobD_generator(query, id):
    message = []
    history = pd.read_csv("open_chat_history.csv")
    columns = history['id']

    if id in columns:
        chat_history = get_chat_by_id(1, csv_file="open_chat_history.csv")
        for msg in chat_history:
            message.append(msg)
        new_rol = {"role": "user", "content": query}
        message.append(new_rol)
    elif not id in columns:
        message.append(system_message)
        new_rol = {"role": "user", "content": query}
        message.append(new_rol)

    #print(message)
        response = job_description_bot(message)
        content = response.choices[0].message.content
        if response.choices[0].message.content:
            print(response.choices[0].message.content)
            message.append({"role":"assistant","content":response.choices[0].message.content})
            store_chat_history(chat_messages=message, chat_id=id)
            return response.choices[0].message.content