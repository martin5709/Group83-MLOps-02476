import requests

url = "http://127.0.0.1:8000/generate"
review = "Lav mig et billede"
response = requests.post(url, json={"request": review})

if response.status_code == 200:
    
    with open("generated_image.png", "wb") as f:
        f.write(response.content)
    print("Image saved as generated_image.png")
else:
    print(f"Failed to generate image. Status code: {response.status_code}")
    print(response.text)