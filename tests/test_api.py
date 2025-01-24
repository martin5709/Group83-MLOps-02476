import requests
import os

def test_api():
    url = "https://api-image-5307485050.europe-west1.run.app/generate"
    request_json = "Lav mig et pænt billede, tak :D"
    response = requests.post(url, json={"request": request_json})
    assert response.status_code == 200 , f'Tried to access the api; however, recived the error {response.status_code}'

    with open("generated_image.png", "wb") as f:
        f.write(response.content)
    assert os.path.exists("generated_image.png"), "The generated_image.png file was not saved"

if __name__ == "__main__":
    test_api()
