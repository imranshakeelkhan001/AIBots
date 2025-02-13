# import requests
#
# url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobPost/getAllJobs"
#
# payload = {}
# headers = {
#   'accept': 'text/plain'
# }
#
# response = requests.request("GET", url, headers=headers, data=payload)
#
# print(response.text)


# import requests
# import json
# def list_of_all_jobs():
#     url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobPost/getAllJobs"
#
#     headers = {
#       'accept': 'text/plain'
#     }
#
#     response = requests.get(url, headers=headers)
#
#     if response.status_code == 200:
#         jobs = response.json()  # Parse JSON response
#         job_titles = [job["title"] for job in jobs]  # Extract job titles
#         print("here is jobs",job_titles)
#         return job_titles
#         # Print each job title on a new line
#         # for title in job_titles:
#         #     print("here is jobs",title)
#     else:
#         print("Failed to fetch jobs:", response.status_code)
#
#
# list_of_all_jobs()

#
# import requests
# import json
# def list_of_selected_candidates():
#     url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobOffer/candidateList"
#
#     payload = {}
#     headers = {
#       'accept': 'text/plain'
#
#     }
#
#     response = requests.get(url, headers=headers)
#
#     # Parse the response JSON
#     candidates = response.json()
#
#     # Extract only name and email
#     filtered_candidates = [{candidate["fullName"]: candidate["candidateEmail"]} for candidate in candidates]
#
#     # Convert to JSON format and print
#     print(json.dumps(filtered_candidates, indent=2))
#     return json.dumps(filtered_candidates, indent=2)

#
# import base64
#
#
# def pdf_to_base64(file_path):
#     """
#     Convert a PDF file to a Base64-encoded string.
#
#     :param file_path: Path to the PDF file.
#     :return: Base64 encoded string.
#     """
#     try:
#         with open(file_path, "rb") as pdf_file:
#             encoded_string = base64.b64encode(pdf_file.read()).decode("utf-8")
#         return encoded_string
#     except FileNotFoundError:
#         print("Error: File not found!")
#         return None
#     except Exception as e:
#         print(f"Error: {e}")
#         return None
#
# base64_pdf =     pdf_to_base64("CVABDULLAHMAHSUD.pdf")
#
# print(base64_pdf)
#

#
# import psycopg2
# import csv
#
# def extract_applicant_table():
#     # Database connection details
#     HOST = "pimsdbsvr.postgres.database.azure.com"
#     PORT = 5432
#     USER = "dbuser"
#     PASSWORD = "Pass@word11"
#     DATABASE = "postgres"
#
#     try:
#         # Connect to the PostgreSQL database
#         connection = psycopg2.connect(
#             host=HOST,
#             port=PORT,
#             user=USER,
#             password=PASSWORD,
#             database=DATABASE
#         )
#         print("✅ Connection successful!")
#
#         # Create a cursor to execute queries
#         cursor = connection.cursor()
#
#         # ✅ Corrected query using "resume" table instead of "ResumeRanked"
#         query = '''
#             SELECT
#                 r.*,
#                 COALESCE(jd."JobTitle", 'NOT FOUND') AS "JobTitle"
#             FROM
#                 "Resume" r
#             LEFT JOIN
#                 "JobDescription" jd
#             ON
#                 r."JDId" = jd."Id";
#         '''
#         cursor.execute(query)
#
#         # Fetch all rows
#         rows = cursor.fetchall()
#
#         # Get column names (for CSV header)
#         column_names = [desc[0] for desc in cursor.description]
#
#         # Save the result to a CSV file
#         with open('ranked_resumes_with_job_titles.csv', mode='w', newline='', encoding='utf-8') as file:
#             writer = csv.writer(file)
#             writer.writerow(column_names)  # Write header
#             writer.writerows(rows)  # Write data
#
#         print("✅ Data has been written to 'ranked_resumes_with_job_titles.csv'.")
#
#         # Check for missing job titles
#         missing_job_titles_count = sum(1 for row in rows if row[column_names.index("JobTitle")] == 'NOT FOUND')
#         if missing_job_titles_count > 0:
#             print(f"⚠ Warning: {missing_job_titles_count} rows have missing job titles (JDId not found in JobDescription).")
#
#         # Close the cursor and connection
#         cursor.close()
#         connection.close()
#         print("✅ Connection closed!")
#
#     except psycopg2.Error as e:
#         print(f"❌ An error occurred: {e}")
#
#
#
# extract_applicant_table()
#
#
import json
import requests
# def get_jdid(title):
#     # API URL
#     url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobPost/getAllJobs"
#
#     # Request headers
#     headers = {
#         'accept': 'text/plain'
#     }
#
#     # Make the API request
#     response = requests.get(url, headers=headers)
#
#     # Parse the response as JSON
#     data = response.json()
#
#     # Extract jdId and title as key-value pairs
#     jdid_title_dict = {item['title']: item['jdId'] for item in data}
#
#     jd_id = jdid_title_dict.get(title)
#     print(jd_id)
#     return jd_id
#
# get_jdid("Chemical Engineer")


# def list_of_selected_candidates():
#     url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobOffer/candidateList"
#
#     payload = {}
#     headers = {
#         'accept': 'text/plain'
#     }
#
#     response = requests.get(url, headers=headers)
#
#     # Parse the response JSON
#     candidates = response.json()
#
#     # Extract only name and email
#     filtered_candidates = [{candidate["fullName"]: candidate["candidateEmail"]} for candidate in candidates]
#
#     # Convert to JSON format and print
#     print(json.dumps(filtered_candidates, indent=2))
#     return json.dumps(filtered_candidates, indent=2)
# list_of_selected_candidates()
#
# import requests
# import json


import requests
import json

#
# def list_of_selected_candidates():
#     url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobOffer/candidateList"
#
#     headers = {
#         'accept': 'text/plain'
#     }
#
#     try:
#         response = requests.get(url, headers=headers)
#
#         # Check if response status is OK (200)
#         if response.status_code != 200:
#             print(f"Error: Received status code {response.status_code}")
#             result = json.dumps({"error": f"Failed to fetch data, status: {response.status_code}"})
#             print("Returning:", result)
#             return result
#
#         # Check if response content is empty
#         if not response.text.strip():
#             print("Error: Received empty response")
#             result = json.dumps({"error": "Empty response from API"})
#             print("Returning:", result)
#             return result
#
#         # Try to parse JSON response
#         try:
#             candidates = response.json()
#         except json.JSONDecodeError:
#             print("Error: Failed to decode JSON")
#             result = json.dumps({"error": "Invalid JSON response from API"})
#             print("Returning:", result)
#             return result
#
#         # Extract only name and email
#         filtered_candidates = [{candidate["fullName"]: candidate["candidateEmail"]} for candidate in candidates]
#
#         # Convert to JSON format
#         result = json.dumps(filtered_candidates, indent=2)
#         print("Returning:", result)  # Print before returning
#         return result
#
#     except requests.exceptions.RequestException as e:
#         print(f"Network error: {e}")
#         result = json.dumps({"error": "Network error while fetching data"})
#         print("Returning:", result)
#         return result
#
#

# list_of_selected_candidates()
import requests

# def list_of_selected_candidates():
#     import requests
#
#     url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobOffer/candidateList"
#
#     payload = {}
#     headers = {
#         'accept': 'text/plain'
#     }
#
#     response = requests.request("GET", url, headers=headers, data=payload)
#
#     # Parse the JSON response
#     candidates = response.json()
#
#     # Extract required fields and format the output
#     formatted_output = []
#
#     for candidate in candidates:
#         candidate_info = {
#             candidate["fullName"].lower(): candidate["candidateEmail"],
#             "jobtitle": candidate["jobTitle"].lower()
#         }
#         formatted_output.append(candidate_info)
#
#     # Print the formatted output
#     print(formatted_output)
#
#
# list_of_selected_candidates()

from openai import OpenAI

client = OpenAI()


def generate_bash_script(history):
    system_message = {
        "role": "system",
        "content": (
            "you are expert AI assistant. you are provided with chat. you will provide reasoning how job description created."
        )
    }

    user_message = {
        "role": "user",
        "content": history
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        reasoning_effort="medium",
        messages=[system_message, user_message]
    )

    return response.choices[0].message.content  # Returning the generated Bash script


generated_script = generate_bash_script()
print(generated_script)
