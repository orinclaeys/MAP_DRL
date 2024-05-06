import requests, json
from DRL import Agent, State

agent = Agent()
agent.loadWeights()

def getState():
    response = requests.get('http://143.129.82.243:5100/Snapshot')
    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print("Error:"+response.status_code)

def sendDecision(decision):
    url = 'https://143.129.82.243:5100/Decision'
    data = {
        "node": decision,
        "cpu" : 0,
        "memory" : 0,
        "storage" : 0,
        "latency" : 0,
        "power" : 0,
        "latitude" : 0,
        "longitude" : 0,
        "timestamp" : 0,
        "timeStampReceived" : 0
    }
    json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json_data, headers=headers)
    if response.status_code == 200:
        print(response.text)
    else:
        print('Error:', response.status_code)








while(True):
    state = getState()
    decision, own = agent.select_action(State)
    sendDecision(decision)