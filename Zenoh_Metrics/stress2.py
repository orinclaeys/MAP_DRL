import time
import os

file_path = "locust_state.txt"

def clear_screen():
    # Clear screen for Windows and Unix/Linux
    os.system('cls' if os.name == 'nt' else 'clear')

def display_file_content():
    try:
        with open(file_path, 'r') as file:
            content = file.read().split(',')
            last_client = content[0]
            last_change_time = float(content[1])

            current_time = time.time()
            seconds_since_last_change = current_time - last_change_time

            clear_screen()  # Clear the screen
            print(f"Last Client: {last_client}")
            print(f"Last Change Time: {last_change_time}")
            print(f"Seconds since last change: {seconds_since_last_change}")

    except FileNotFoundError:
        clear_screen()  # Clear the screen
        print(f"File '{file_path}' not found.")
    except Exception as e:
        clear_screen()  # Clear the screen
        print(f"An error occurred: {e}")

while True:
    display_file_content()
    time.sleep(1)
