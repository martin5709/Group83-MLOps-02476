import requests

url = "https://data-drift-report-generator-5307485050.europe-west1.run.app/report"
n = 10
response = requests.get(url, params={"n": n})

if response.status_code == 200:
   # Save the HTML content from the response
   with open("report.html", "w", encoding="utf-8") as f:
       f.write(response.text)
else:
   print(f"Failed to generate report. Status code: {response.status_code}")
