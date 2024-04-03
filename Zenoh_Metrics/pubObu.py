#!/ur/bin/env python3
import ipaddress
import socket
import time
import argparse
import itertools
import json
import zenoh
from zenoh import config
from subprocess import check_output
from subprocess import STDOUT, check_call
import re
import json
from datetime import datetime
import requests
import gpsd


def get_terminal_info():
    # Get the hostname of the machine
    hostname = socket.gethostname()

    # Split the hostname using "-" as the delimiter
    parts = hostname.split("-")

    if len(parts) >= 2:
        # Extract the substring before the hyphen
        terminal_info = parts[1]

        # Take the first 4 characters of the substring
        rsu_name = terminal_info[:4]

        return rsu_name

    # Return a default value or handle the case when the hostname is not in the expected format
    return "Terminal information not found"


def get_local_ip_addresses():
    # Get a list of all IP addresses associated with the current host
    host_name = socket.gethostname()
    local_ip_addresses = socket.gethostbyname_ex(host_name)[2]

    return local_ip_addresses


def find_matching_ip(ip_list, target_mask):
    matching_ips = []

    # Convert the target mask to an IPv4 network object
    target_network = ipaddress.IPv4Network(target_mask, strict=False)

    # Iterate through the list of IP addresses and check if they match the target network
    for ip_str in ip_list:
        ip = ipaddress.IPv4Address(ip_str)
        if ip in target_network:
            matching_ips.append(ip_str)

    return matching_ips


def get_Ip():
    # Get all local IP addresses of the current host
    local_ip_addresses = get_local_ip_addresses()

    # Example usage
    target_mask = "143.129.82.0/24"

    matching_ips = find_matching_ip(local_ip_addresses, target_mask)
    return f"{matching_ips[0]}" if matching_ips else ""


def extract(Lines):
    row_content = Lines.split()
    cpu = int(row_content[7].strip("%"))
    memory = int(row_content[9].strip("%"))
    return cpu, memory


def extract_power_value(input_string):
    pattern = r'INTEGER:\s*(\d+)'
    match = re.search(pattern, input_string)
    if match:
        return int(match.group(1))
    else:
        return None


def extract_location():
    current_location = gpsd.get_current()
    return current_location.lat, current_location.lon


if __name__ == "__main__":
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
                        default='demo/test/obu',
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
    session = zenoh.open(conf)

    print(f"Declaring Publisher on '{key}'...")
    pub = session.declare_publisher(key)
    host = get_terminal_info()
    gpsd.connect(f'smarthw-{host}gpcu.idlab.uantwerpen.be')

    for idx in itertools.count() if args.iter is None else range(args.iter):
        dt = datetime.now()
        ts = datetime.timestamp(dt)
        ip = get_Ip()
        latitude, longitude = extract_location()
        req1 = requests.get("http://143.129.82.117:5002/")
        req2 = requests.get("http://143.129.82.113:5002/")
        req3 = requests.get("http://143.129.82.129:5002/")

        latency1 = req1.elapsed.total_seconds()*1000
        latency2 = req2.elapsed.total_seconds()*1000
        latency3 = req3.elapsed.total_seconds()*1000

        dictionary = {"metrics": [{"hostname": "rsu4", "metric": "latency", "value": latency1},
                                  {"hostname": "rsu6", "metric": "latency", "value": latency2},
                                  {"hostname": "rsu7", "metric": "latency", "value": latency3},
                                  {"hostname": f"{host}", "metric": "location", "value": {"latitude": latitude, "longitude": longitude}}
                                  ],
                      "timestamp": ts,
                      "origin": {"ip": ip, "host": host}}

        json_object = json.dumps(dictionary, indent=4)
        time.sleep(1)
        buf = f"{json_object}"
        print(buf)
        pub.put(buf)

    pub.undeclare()
    session.close()
