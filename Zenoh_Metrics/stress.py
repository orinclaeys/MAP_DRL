from locust import HttpUser, task, constant
import random
import time
import os

class WebsiteTestUser(HttpUser):
    wait_time = constant(0.1)
    file_path = "locust_state.txt"

    def on_start(self):
        """ on_start is called when a Locust start before any task is scheduled """
        self.clientList = ["http://143.129.82.117:5002/", "http://143.129.82.113:5002/", "http://143.129.82.129:5002/"]
        
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                content = file.read().split(',')
                self.last_client = content[0]
                self.last_change_time = float(content[1])
        else:
            self.last_client = random.choice(self.clientList)
            self.last_change_time = time.time()
            self._write_state_to_file()

    def on_stop(self):
        """ on_stop is called when the TaskSet is stopping """
        pass

    def _write_state_to_file(self):
        with open(self.file_path, 'w') as file:
            file.write(f"{self.last_client},{self.last_change_time}")

    @task(3)
    def hello_world(self):
        with open(self.file_path, 'r') as file:
            content = file.read().split(',')
            self.last_client = content[0]
            self.last_change_time = float(content[1])
        
        if time.time() - self.last_change_time > 80:
            self.last_client = random.choice(self.clientList)
            self.last_change_time = time.time()
            self._write_state_to_file()

        self.client.get(self.last_client)
