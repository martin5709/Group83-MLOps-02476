from locust import HttpUser, task, between


# Written 100% by ChatGPT...
class ImageAPILoadTest(HttpUser):
    # Wait time between tasks (min=1s, max=3s)
    wait_time = between(1, 3)

    @task
    def generate_image(self):

        payload = {"request": "Please make me an image :D"}
        headers = {"Content-Type": "application/json"}

        with self.client.post(
            url="/generate",
            json=payload,
            headers=headers,
            catch_response=True,
        ) as response:

            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code {response.status_code}")

