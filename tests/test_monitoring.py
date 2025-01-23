#import requests

#url = "http://127.0.0.1:8000/report"
#n = 10
#response = requests.post(url, json={"n":n})
# response = requests.post(url, json={"n": n})

#if response.status_code == 200:
#    # Save the HTML content from the response
#    with open("report.html", "w", encoding="utf-8") as f:
#        f.write(response.text)
#else:
#    print(f"Failed to generate report. Status code: {response.status_code}")
