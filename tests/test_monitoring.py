import requests
import os

def test_monitoring():
    url = "https://data-drift-report-generator-5307485050.europe-west1.run.app/report"
    n = 10
    response = requests.get(url, params={"n": n})

    assert response.status_code == 200 , f'Tried to access report; however, recived the error {response.status_code}'

    with open("report.html", "w", encoding="utf-8") as f:
        f.write(response.text)
    assert os.path.exists("report.html"), "The report.html file was not saved"

if __name__ == "__main__":
    test_monitoring()
