# import requests
# import json
#
# client_id = 'ybrhxu41wwx4iz3i'
# client_secret = 'a0K530FQ'
# scope = 'emsi_open'
#
# def access_gen(client_id, client_secret, scope):
#     url = "https://auth.emsicloud.com/connect/token"
#     payload = f"client_id={client_id}&client_secret={client_secret}&grant_type=client_credentials&scope={scope}"
#     headers = {"Content-Type": "application/x-www-form-urlencoded"}
#
#     response = requests.post(url, data=payload, headers=headers)
#
#
#     new_response=json.loads(response.text)['access_token']
#     # print(new_response)
#     return new_response
#
# token = access_gen(client_id, client_secret, scope)
#
#
# def update_env_key(file_path, key, new_value):
#     try:
#         with open(file_path, "r") as file:
#             lines = file.readlines()
#     except FileNotFoundError:
#         lines = []
#
#     with open(file_path, "w") as file:
#         found = False
#         for line in lines:
#             if line.startswith(f"{key}="):
#                 file.write(f"{key}=\"{new_value}\"\n")
#                 found = True
#             else:
#                 file.write(line)
#         if not found:
#             file.write(f"{key}=\"{new_value}\"\n")
#
# # Example usage
# update_env_key("../.env", "access_token", token)
#
#
#
#
#
