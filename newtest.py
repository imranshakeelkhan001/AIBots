import requests
import json
def ad_resume(email,file):
    url = "http://af8f9979b16504831a400599cb562ea7-2055552796.eu-north-1.elb.amazonaws.com:8080/JobPost/addResumeFileBytes"

    payload = json.dumps({
      "email": str(email),
      "fileBase64": f"{file}"
    })
    headers = {
      'accept': 'text/plain',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)