from locust import task
from locust import between
from locust import HttpUser

sample = {
  "team": "portugal",
  "opposition": "ghana",
  "minute_taken": 64,
  "scoreline": "draw",
  "venue": "neutral",
  "previous_penalty": 1,
  "natural_foot": "right",
  "position_style": "left",
  "competition": "wc",
  "gk": "ati_zigi"
}

class MLZoomUser(HttpUser):
    """
    Usage:
        Start locust load testing client with:
            locust -H http://localhost:3000
        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    @task
    def classify(self):
        self.client.post("/classify", json=sample)

    wait_time = between(0.01, 2)