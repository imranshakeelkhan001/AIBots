import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import pandas as pd
import json

load_dotenv()
client = OpenAI()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


from datetime import datetime

def current_date():
    return datetime.now().strftime("%d %B %Y")

class CalendarEvent(BaseModel):
    feed_back: bool
    offer_letter:bool
    offer_send: bool








def canvas_flag(history):
    history[0] = {"role": "system", "content": "feed_back: Make sure to turn this flag True, When you ask from user 'how was the feed back for the candidate?'"
                                               "offer_letter: Turn this flag True when you generate the complete offer letter."
                                               "offer_send: Turn this flag True when user says that offer letter is approved from my side."}

    print(history)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
         messages= history,
        #     {"role": "system", "content": "Extract  a boolean flag from the given chat that if the assistant generated any Job description or not"},
        #     {"role": "user", "content": f"{history[1:]}"},
        # ],
        response_format=CalendarEvent,
    )

    event = completion.choices[0].message.parsed
    return event

# selected_candidates = ["azeem","osama"]

class str_data(BaseModel):
    job_title:str
    salary:str
    working_hours:str
    location:str
    name:str
    email:str
    description_message:str
    benefits:str
    employment_terms:str
    joining_date: str
    offer_expiry_date:str
    offer_acceptance:str




# Function to extract skills from the job description using OpenAI
def offer_data(message):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": f""" you are given with a chat of user and assistant. the assistant generated the offer letter based on user queries. your job is to extract the parameters from it  please note that and make  sure that The default value of every parameter is 'empty'..
        please note that and make  sure that The default value of every parameter is 'empty'.
    f"Your task is to extract following information for the candidate."
        f"1. name"
        f"2. job_title"
        f"3. salary" e.g 85000 along with the currency e.g 85000 PKR
        f"4. working_hours"
        f"5. location"
        f"6. description_message"
        f"7  "email"
        f"8. benefits"
        f"9. employment_terms"
        f"10. joining_date"
        f"11. offer_expiry_date"
        f"12. offer_acceptance" e.g to accept the offer please sign and return a copy of this letter by 'offer_expiry_date'
         
        Please note that  do not provide any parameter response on your own.
        If any of the above data is not found than leave it as 'empty'."""},

                  {"role": "user", "content": message}],
        response_format=str_data,
    )

    if not completion.choices or not completion.choices[0].message.parsed:
        print("No valid offer data extracted.")
        return ["empty"] * 12  # Return 'empty' for all parameters if no response

    parsed_data = completion.choices[0].message.parsed

    return (
        getattr(parsed_data, "name", "empty"),
        getattr(parsed_data, "job_title", "empty"),
        getattr(parsed_data, "salary", "empty"),
        getattr(parsed_data, "working_hours", "empty"),
        getattr(parsed_data, "location", "empty"),
        getattr(parsed_data, "description_message", "empty"),
        getattr(parsed_data, "email","empty"),
        getattr(parsed_data, "benefits", "empty"),
        getattr(parsed_data, "employment_terms", "empty"),
        getattr(parsed_data, "joining_date", "empty"),
        getattr(parsed_data, "offer_expiry_date", "empty"),
        getattr(parsed_data,"offer_acceptance","empty")
    )

# def list_of_selected_candidates():
#     print("hello done")
#     return ['azeem@gmail.com','osama@gmail.com']


def list_of_selected_candidates():
    import requests

    url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobOffer/candidateList"

    payload = {}
    headers = {
        'accept': 'text/plain'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    # Parse the JSON response
    candidates = response.json()

    # Extract required fields and format the output
    formatted_output = []

    for candidate in candidates:
        candidate_info = {
            candidate["fullName"].lower(): candidate["candidateEmail"],
            "jobtitle": candidate["jobTitle"].lower()
        }
        formatted_output.append(candidate_info)

    # Print the formatted output
    print(formatted_output)
    return formatted_output

# import requests
# import json

# def list_of_selected_candidates():
#     url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobOffer/candidateList"
#
#     headers = {'accept': 'text/plain'}
#
#     try:
#         response = requests.get(url, headers=headers)
#
#         # Check if response status is OK (200)
#         if response.status_code != 200:
#             print(f"API Error: Received status code {response.status_code}")
#             return json.dumps({"error": f"API Error: Status {response.status_code}"})
#
#         # Check if response content is empty
#         if not response.text.strip():
#             print(" Warning: API returned an empty response.")
#             return json.dumps({"error": "API returned an empty response"})
#
#         # Try to parse JSON response
#         try:
#             candidates = response.json()
#         except json.JSONDecodeError:
#             print(f"JSONDecodeError: Invalid JSON received from API. Response content: {response.text}")
#             return json.dumps({"error": "Invalid JSON response from API"})
#
#         # Extract only name and email
#         filtered_candidates = [{candidate["fullName"]: candidate["candidateEmail"]} for candidate in candidates]
#
#         # Convert to JSON format and print
#         result = json.dumps(filtered_candidates, indent=2)
#         print("Returning:", result)  # Print before returning
#         return result
#
#     except requests.exceptions.RequestException as e:
#         print(f" Network error: {e}")
#         return json.dumps({"error": "Network error while fetching data"})



def store_chat_history(chat_messages, chat_id, csv_file="offer_chat_history.csv"):
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
        chat_df = pd.DataFrame(columns=["id", "offer_chat_history"])

    # Check if the ID exists in the DataFrame
    if chat_id in chat_df["id"].values:
        # Update the existing row
        chat_df.loc[chat_df["id"] == chat_id, "offer_chat_history"] = str(chat_messages)
    else:
        # Add a new row
        new_row = {"id": chat_id, "offer_chat_history": str(chat_messages)}
        chat_df = pd.concat([chat_df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    chat_df.to_csv(csv_file, index=False)
    print(f"Chat history for ID {chat_id} has been saved to {csv_file}.")
    #print(f"Chat history for ID {chat_id} has been saved to {csv_file}.")









# Initial system message
system_message = {
        "role": "system",
        "content": (
            "You are a helpful DESCON HR assistant.Start conversation with greetings by introducing your name only 1 time ' I am Descon HR Assistant'"
            "your job is to generate the offer letter."
            "Please make sure Do not ask multiple questions in 1 response. always greet user when you receive response from user. at start of conversation you only greet user first. than interact with user by asking  each question should be connected to another."
            "ask user about the name of the candidate"
            f"once you have name of the candidate. you will match it with the name in the list of the  selected candidates, if the name matched with any email than you will must show it user and ask him 'this is the email of the candidate can you confirm it? please note do not include numbering in your response."
            f"after that make sure that  you will provide the job title that is associated with the name of the candidate and ask user ' can you please confirm this is the job position for the candidate right?' "
            f"if the name that user provide is not matched with any name of the email you will not entertain the offer letter. you will respond that this name is not among the selected candidates, Please provide the selected candidate name."
            f"if user says show me list of selected candidates, you will show him the list of selected candidates by showing the emails along with their names. please note do not include numbering in your response. "
            f"after that if user provide the name, you must confirm the provided name email by showing it to user e.g can you please confirm this email is associated with the user name etc is it correct?"
            f"please make sure that 'if user enter a name and the name matched with more than 1 emails you will confirm from user which one is the email of this candidate by providing him the matched emails with the name.please note do not include numbering in your response. please do not generate emails from your own. use only the provided emails that you have."
            "please make sure that you 'Do not provide or generate any email from your own'."
            "you will only show the email of the candidate that user enter by matching."
 #           f"here is a list of selected candidates {selected_candidates}. if candidate name is in the list than you will proceed further. if name is not in the list you will say that this candidate is not available in selected list."
            "after that you will ask user about the feedback of the candidate. if user respond it was good than you suggest ok great,"
            " or if user respond with it was normal you will suggest him that we can add a probation period for 2  or 3 months so that we can check or analyze him based on his performance."
            "if user did not respond to 'how was the feedback of the candidate', than you move to next. questions."
            "for the job position you will consider the job title that is associated with the candidate name that you already have."
            #"after that ask user about the position for the candidate"
            "after that ask user about the start date of the job"
            "after that ask user about the working hours of the candidate"
            "after that ask user about the salary of the candidate per month"
            "after that ask user about the job location"
            "you will generate a professional Offer Letter  that include all the details in a little depth."
            "you also mentioned in the offer letter that 'to accept this offer, please sign and return a copy of this letter by replying to this email."
            "please note: make sure you will not ask multiple questions in 1 response."
            "if user didn't respond to any question you will suggest him the answer etc "
            "based on this you will generate the offer letter"
            "make sure to ask relevant questions for generating the offer letter."
            "The offer letter consist of headings with 'position details' it include all the details about the position e.g job title , start date, salary,working hours, location."
            "after that The offer letter consist of heading  'Benefits ' health insurance any , paid vacations per year, professional development and training opportunities."
            "after that the offer letter consist of heading 'Employment Terms' e.g that this is full time role etc"
            "after that the offer letter consist of heading 'acceptance' e.g to accept this offer, please sign and return a copy of this letter by replying to this email, if you have any questions or need further clarification, feel free to contact me etc "
            f"once you get all the details from user, also ask about the user for expiry of the offer letter date. use the current date for {current_date()} your reference. "
            f"please make sure to generate the professional offer letter. make sure to include all the details."
            f"Please note that  you will include 2 to 3 lines of role paragraph in offer letter. and it should be professional and aligned with job title."
            "once you generate the offer letter you will ask user to check it or does user want to add anything in it? must use '?' at the end of this question"
            "After providing the offer letter.User may ask you for some changes, try to make these changes specific only in that offer letter that you just provided. Note:  'Do not provide new offer letter' "
            "unless the user want any other category offer letter that user asked and provide the desired output that user want."
            "once user says it is finalized or ok from my side. than you will ask should i proceed to sent him the offer letter?."
            "if user says yes than you will respond that e.g ok sure it will be sent in a while"
            "Do not ask multiple questions in 1 response. "
            "Do not include in offer letter '**[Your Company Letterhead]**"
            "As you are representative of DESCON so add Company DESCON in your offer letter"
            "Please make sure to get all the above information from user by asking relevant questions."

            ""
        )}


# Conversational bot logic
def conversational_Bot(message):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        # tools=tools,
        temperature=0.2

    )
    return completion
#
def get_chat_by_id(chat_id, csv_file="offer_chat_history.csv"):
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
            chat_history_str = history.loc[history["id"] == chat_id, "offer_chat_history"].values[0]

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


async def offer_generator(query, id):
    message = []
    # job = list_of_selected_candidates()
    new_jobs = {"role": "assistant", "content":f"here is list of selected candidates with job titles {str(list_of_selected_candidates())}. if it is empty than there is no selected candidates " }
    print(new_jobs)
    message.append(new_jobs)
    history = pd.read_csv("offer_chat_history.csv")
    columns = history['id']
    if int(id) in list(columns):
        print("inside if")
        chat_history = get_chat_by_id(int(id), csv_file="offer_chat_history.csv")
        for msg in chat_history:
            message.append(msg)
        new_rol = {"role":"user", "content": query}
        message.append(new_rol)
    elif not int(id) in list(columns):
        print("inside elif")
        message.append(system_message)
        new_rol = {"role":"user", "content": query}
        message.append(new_rol)

    response = conversational_Bot(message)
    content = response.choices[0].message.content
    #message.append({"role":"assistant", "content":f"here is a list of selected candidates {list_of_selected_candidates()}"})
    #print(content)

    if content:
        message.append({"role": "assistant", "content": content})


    name,job_title,salary,working_hours,location,description_message,email,benefits,employment_terms,joining_date,offer_expiry_date,offer_acceptance = offer_data(str(message))

    # Store updated chat history
    store_chat_history(message, id)
    #if 'feedback' in query.lower() or 'feed back' in query.lower():


    # Return response with extracted offer details
    return {
        "chat": content,

        "canvas_flag": canvas_flag(message),
        "offer_details": {
            "name": name,
            "job_title": job_title,
            "salary":salary,
            "working_hours":working_hours,
            "location":location,
            "description_message":description_message,
            "email":email,
            "benefits":benefits,
            "employment_terms":employment_terms,
            "joining_date":joining_date,
            "offer_expiry_date":offer_expiry_date,
            "offer_acceptance": offer_acceptance
        }
    }



