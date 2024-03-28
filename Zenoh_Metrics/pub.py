import ipaddress
import socket
import time
import argparse
import itertools
import json
import zenoh
from zenoh import config
import subprocess
from subprocess import check_output
from subprocess import STDOUT, check_call
import re
import json
from datetime import datetime
import gpsd


def get_terminal_info():
    # Get the hostname of the machine
    hostname = socket.gethostname().split("-")

    if len(hostname) >= 2:        
        return hostname[1][:4]
       
    return "Terminal information not found"

def get_Ip():
    # Get all local IP addresses of the current host
    local_ip_addresses = socket.gethostbyname_ex(socket.gethostname())[2]
    target_mask = "143.129.82.0/24"
    matching_ips = []

     # Convert the target mask to an IPv4 network object
    target_network = ipaddress.IPv4Network(target_mask, strict=False)

    # Iterate through the list of IP addresses and check if they match the target network
    for ip_str in local_ip_addresses:
        ip = ipaddress.IPv4Address(ip_str)
        if ip in target_network:
            matching_ips.append(ip_str)

    return f"{matching_ips[0]}" if matching_ips else ""


def getNodeMetrics():    
    row_content = subprocess.getoutput('kubectl top nodes').split()
    cpu = int(row_content[7].strip("%"))
    memory = int(row_content[9].strip("%"))
    storage = subprocess.check_output(['df', '-h', '/']).decode('utf-8').split('\n')[1].split()[3][:-1]
    return cpu, memory, storage


def getPowerMetrics():
    power_output = subprocess.getoutput(
        f'snmpget -c private -v 1 smarthw-{get_terminal_info()}pdu.idlab.uantwerpen.be .1.3.6.1.4.1.34097.3.1.1.4.1')
    pattern = r'INTEGER:\s*(\d+)'
    match = re.search(pattern, power_output)
    if match:
        return int(match.group(1))
    else:
        return None


def getNodeLocation():
    current_location = gpsd.get_current()
    return current_location.lat, current_location.lon


# --- Command line argument parsing --- --- --- --- --- ---
parser = argparse.ArgumentParser(
    prog='z_pub',
    description='zenoh pub example')
parser.add_argument('--mode', '-m', dest='mode',
                    choices=['peer', 'client'],
                    type=str,
                    help='The zenoh session mode.')
parser.add_argument('--connect', '-e', dest='connect',
                    metavar='ENDPOINT',
                    action='append',
                    type=str,
                    help='Endpoints to connect to.')
parser.add_argument('--listen', '-l', dest='listen',
                    metavar='ENDPOINT',
                    action='append',
                    type=str,
                    help='Endpoints to listen on.')
parser.add_argument('--key', '-k', dest='key',
                    default=f'demo/test/{get_terminal_info()}',
                    type=str,
                    help='The key expression to publish onto.')
parser.add_argument('--value', '-v', dest='value',
                    default='Pub from Python!',
                    type=str,
                    help='The value to publish.')
parser.add_argument("--iter", dest="iter", type=int,
                    help="How many puts to perform")
parser.add_argument('--config', '-c', dest='config',
                    metavar='FILE',
                    type=str,
                    help='A configuration file.')

args = parser.parse_args()
conf = zenoh.Config.from_file(
    args.config) if args.config is not None else zenoh.Config()
if args.mode is not None:
    conf.insert_json5(zenoh.config.MODE_KEY, json.dumps(args.mode))
if args.connect is not None:
    conf.insert_json5(zenoh.config.CONNECT_KEY, json.dumps(args.connect))
if args.listen is not None:
    conf.insert_json5(zenoh.config.LISTEN_KEY, json.dumps(args.listen))
key = args.key
value = args.value

# initiate logging
zenoh.init_logger()

print("Opening session...")
# conf = zenoh.Config().from_file("zenoh_config.json5")
session = zenoh.open(conf)
# session = zenoh.open(conf)

print(f"Declaring Publisher on '{key}'...")
pub = session.declare_publisher(key)
host = get_terminal_info()
gpsd.connect(f'smarthw-{host}gpcu.idlab.uantwerpen.be')

for idx in itertools.count() if args.iter is None else range(args.iter):
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    ip = get_Ip()    
    latitude, longitude = getNodeLocation()
    cpu, memory, storage = getNodeMetrics()
    power = getPowerMetrics()
    dictionary = {"metrics":
                  [{"value": {"metric_value": {"latitude": latitude, "longitude": longitude}, "timestamp": ts, "unit": "F"}, "metric": "location"},
                   {"value": {"metric_value": memory, "timestamp": ts, "unit": "%"}, "metric": "memory"},
                   {"value": {"metric_value": cpu, "timestamp": ts, "unit": "%"}, "metric": "cpu"},
                   {"value": {"metric_value": storage, "timestamp": ts, "unit": "%"}, "metric": "storage"},
                   {"value": {"metric_value": power, "timestamp": ts, "unit": "W"}, "metric": "power"}],
                  "origin": {"ip": ip, "host": host}}
    json_object = json.dumps(dictionary, indent=4)
    time.sleep(1)
    buf = f"{json_object}"
    print(buf)
    pub.put(buf)

pub.undeclare()
session.close()
