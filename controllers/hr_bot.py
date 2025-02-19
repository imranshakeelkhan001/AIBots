from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
import psycopg2
import csv
import pandas as pd
from openai import OpenAI
import json
import os
from pydantic import BaseModel




def extract_applicant_table():
    # Database connection details
    HOST = "pimsdbsvr.postgres.database.azure.com"
    PORT = 5432
    USER = "dbuser"
    PASSWORD = "Pass@word11"
    DATABASE = "postgres"

    try:
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=DATABASE
        )
        print("✅ Connection successful!")

        # Create a cursor to execute queries
        cursor = connection.cursor()

        # ✅ Corrected query using "resume" table instead of "ResumeRanked"
        query = '''
            SELECT 
                r.*, 
                COALESCE(jd."JobTitle", 'NOT FOUND') AS "JobTitle"
            FROM 
                "Resume" r
            LEFT JOIN 
                "JobDescription" jd
            ON 
                r."JDId" = jd."Id";
        '''
        cursor.execute(query)

        # Fetch all rows
        rows = cursor.fetchall()

        # Get column names (for CSV header)
        column_names = [desc[0] for desc in cursor.description]

        # Save the result to a CSV file
        with open('ranked_resumes_with_job_titles.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(column_names)  # Write header
            writer.writerows(rows)  # Write data

        print("✅ Data has been written to 'ranked_resumes_with_job_titles.csv'.")

        # Check for missing job titles
        missing_job_titles_count = sum(1 for row in rows if row[column_names.index("JobTitle")] == 'NOT FOUND')
        if missing_job_titles_count > 0:
            print(f"⚠ Warning: {missing_job_titles_count} rows have missing job titles (JDId not found in JobDescription).")

        # Close the cursor and connection
        cursor.close()
        connection.close()
        print("✅ Connection closed!")

    except psycopg2.Error as e:
        print(f"❌ An error occurred: {e}")

def store_chat_history(chat_messages, chat_id, csv_file="hr_chat_history.csv"):
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


def run_csv_query(query):
    """
    Function to create and run a CSV agent with a given query.
    """
    # Initialize the OpenAI LLM
    llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)

    # Create the CSV agent
    csv_agent = create_csv_agent(
        llm,
        "ranked_resumes_with_job_titles.csv",  # Replace with your CSV file path
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        allow_dangerous_code=True
    )

    # Run the query through the CSV agent
    csv_response = csv_agent.run(query + str("""must give me their JDid and Id in following example format (must use this format): 
                                            user : Give me top 2 score  AI engineers
                                            AI Assistant:

                                            [{"JDId": [56, 56], "Id": [26, 27]}]  must use this format
                                            Note: If no matching candidate found then you can reply in english sentence, use double quote with keys in json because your reponse will be used in json.loads() function"""))
    return csv_response


client = OpenAI()


def get_chat_by_id(chat_id, csv_file="hr_chat_history.csv"):
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


class response_format(BaseModel):
    appointment_flag: bool
    llm_response: str
    candidate_email_address: list[str]
    interviewers_email_address: list[str]
    candidate_info: str
    show_posted_jobs: bool
    JDid: list[int]
    id: list[int]


def hr_assistant(message):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=message,
        response_format=response_format,
    )

    event = completion.choices[0].message.parsed
    return event


system_message = {
    "role": "system",
    "content": "You are a conversation AI assistant at DESCON, your task is to assist HR analyst to shortlist the best candidates for the job. "
               "llm_response: this variable holds the responses of LLM (in this case it is you). The responses should be conversational. Whenever someone greets you then greet them with asking how can i help you with candidate shortlisting?"

               "interviewers_email_address: before setting the appointment_flag as True you must ask the user about the email address of interviewer"
               "candidate_email_address: this is the email address of candidate"
               "appointment_flag: this flag (when True) will be used to start appointment booking. turn this flag variable as true when you ask the user if he wants to book appointment. Must ask him before setting it True. it can only be true when you have asked "
               "the user about interviewers_email_address and candidate_email_address. Make sure It can only be true when both emails are provided by user."
               "candidate_info: example of this is  'top 2 AI Engineers', 'top 7 Marketing Head' this will be used by an AI Agent , anticipate from the chat if the user's last message is to get the details of candidate like, give me top 3 AI engineers,please note in this string there must be job title in its query, "
               "if the user does not provide job title then ask him the job title whose candidates to be filtered, if ask if not any job title found in query like 'ai engineer', 'content writer' etc. Note: if the user lasts message is not a query/question about a candidate than leave this as empty string."
               "leave this as empty string. if user asking to schedule an interview with a candidate than leave candidate_info this as empty string"
               "leave JDid and id as 0, its not for your use"
               "show_posted_jobs: this flag is only true when user says 'show me posted jobs' or something similar to that otherwise it remains false throughout the conversation'"
               "Note: the llm_response is your response which the user will see and interact with you with that, so keep the conversational flow throughout, "
               "Step1: the user will ask for top n candidates then you will give the query to candidate_info:  and after that you will ask the user if he want to book interview with any one of them"
               "if the user want to book interview then firstly ask him to input interviewers_email_address and then  candidate candidate_email_address, when you have gotten the both emails then in llm response say thankyou for providing the details by "
               "turning the appointment flag true. turn the appointment flag true when you got both the email addresses. When user asks to schedule interview then must make the candidate_info an empty string again."
               "Note: just greet the user once in the beginning and not after that. if the user says schedule interview/appointment with example@email.com then consider it as candidate_email_address and donot ask from user from it."}


class EventData(BaseModel):
    JDId: list[int]
    Id: list[int]


def convert_string_to_json(input_string: str) -> EventData:
    # First, parse the string into a Python object using OpenAI's model
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system",
             "content": "extract the JDid and id from the given context. If no ids found leave it as its "},
            {"role": "user", "content": input_string},
        ],
        response_format=EventData,  # Ensuring the output is in the format of EventData
    )
    JDid = completion.choices[0].message.parsed.JDId
    Id = completion.choices[0].message.parsed.Id
    # Extract the parsed response as a Python object
    return JDid, Id

# Define a dictionary to store the mappings
def save_jdid(id, jdid):

    try:
        df = pd.read_csv("jdid.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["id", "jdid"])

    if id in df["id"].values:
        df.loc[df["id"] == id, "jdid"] = jdid  # Update existing entry
        print(f"Updated: ID {id} -> JDID {jdid}")
    else:
        new_row = pd.DataFrame({"id": [id], "jdid": [jdid]})
        df = pd.concat([df, new_row], ignore_index=True)
        print(f"Saved: ID {id} -> JDID {jdid}")
    df.to_csv("jdid.csv", index=False)

def get_jdid_from_csv(id):

    id = str(id)

    try:
        df = pd.read_csv("jdid.csv", dtype={"id": str})

        row = df[df["id"] == id]

        if not row.empty:
            return row["jdid"].values[0]
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return "Not Found"

    return "Not Found"


async def hr_chatbot(query, id):
    message = []
    history = pd.read_csv("hr_chat_history.csv")
    columns = history['id']

    if int(id) in list(columns):
        print("inside if")
        chat_history = get_chat_by_id(int(id), csv_file="hr_chat_history.csv")
        for msg in chat_history:
            message.append(msg)
        print("here is message",message)
        new_rol = {"role": "user", "content": query}
        message.append(new_rol)
    elif not int(id) in list(columns):
        print("inside elif")
        message.append(system_message)
        new_rol = {"role": "user", "content": query}
        message.append(new_rol)
        hr_assistant(message)
        extract_applicant_table()

    response = hr_assistant(message)
    if response.candidate_info:
        info = run_csv_query(str(response.candidate_info))
        print("agent answers", info)
        if str(info) == 'result':
            response.llm_response = "I can not find any matching candidate"
            return response
        if str(info) == 'final_result':
            response.llm_response = "I can not find any matching candidate"
            return response

        message.append({"role": "assistant", "content": f"{info}"})
        try:
            print("inside try")
            # print(1)
            # response.llm_response = json.dumps(info)
            # response.llm_response = response.llm_response.replace("'", '"')
            response.llm_response = str(info)
            if not isinstance(response.llm_response, list):
                response.llm_response = [response.llm_response]
                print(type(response.llm_response))

            JDid, Id = convert_string_to_json(str(response.llm_response))

            print("The ids are", JDid, Id)
            response.JDid = JDid
            response.id = Id
            save_jdid(id,response.JDid[0])
            if len(JDid) == 0:
                response.llm_response = "Can't find any matching candidate."
            store_chat_history(message, id)
            # response.candidate_info = ""
            return response

        except:
            print("inside except")
            if response.appointment_flag:
                response.JDid = get_jdid_from_csv(id)
            response.llm_response = str(info)
            message.append({"role": "user", "content": response.llm_response})
            store_chat_history(message, id)
            return response
    message.append({"role": "user", "content": response.llm_response})
    response.id = []
    response.JDid = []
    store_chat_history(message, id)
    response.candidate_info = ""
    if response.appointment_flag == True:
        jdid_from_csv = get_jdid_from_csv(id)
        if jdid_from_csv.isdigit():  # Checks if the value contains only numeric characters
            response.JDid = [int(jdid_from_csv)]
        else:
            response.JDid = None  # Or handle it differently based on your logic
            print("Warning: JDID not found in CSV. Setting response.JDid to None.")

    #        response.JDid = [int(get_jdid_from_csv(id))]
    return response
