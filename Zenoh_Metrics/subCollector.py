#
# Copyright (c) 2022 ZettaScale Technology
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0, or the Apache License, Version 2.0
# which is available at https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
#
# Contributors:
#   ZettaScale Zenoh Team, <zenoh@zettascale.tech>
#

import sys
import time
from datetime import datetime
import argparse
import json
import zenoh
from zenoh import Reliability, Sample
import requests
import json

# --- Command line argument parsing --- --- --- --- --- ---
parser = argparse.ArgumentParser(
    prog='z_sub',
    description='zenoh sub example')
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
                    default='demo/**',
                    type=str,
                    help='The key expression to subscribe to.')
parser.add_argument('--config', '-c', dest='config',
                    metavar='FILE',
                    type=str,
                    help='A configuration file.')


def writeToDb(node, topic, value, tmstp):
    data = {topic: value, "node": node}
    with open(f"{node}{topic.upper()}.txt", "a") as f:
        f.write(f"{tmstp} {value}\n")


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

# Zenoh code  --- --- --- --- --- --- --- --- --- --- ---


# initiate logging
zenoh.init_logger()

print("Opening session...")
# conf = zenoh.Config().from_file("zenoh_config.json5")
session = zenoh.open(conf)
# session = zenoh.open(conf)

print("Declaring Subscriber on '{}'...".format(key))


def listener(sample: Sample):
    # print(f">> [Subscriber] Received {sample.kind} ('{sample.key_expr}': '{sample.payload.decode('utf-8')}')")
    received = sample.payload.decode("utf-8")
    received = json.loads(received)
    topic = str(sample.key_expr)
    # sender_host = received['origin']['host']    
    # sender_ip = received['origin']['ip']
    sender_host = received.get('origin', {}).get('host', None)
    sender_ip =  received.get('origin', {}).get('ip', None)

    if topic == "demo/test/diagnostics":
        print(received)
        diagnostics = received["diagnosis"]["host"]
        print(diagnostics)
        # '{"diagnosis": {"client": "client2", "host": "rsu5", "status": "non-healthy"}}'
        if diagnostics == "rsu4":
            data = [1, 0, 0, 0]
        elif diagnostics == "rsu5":
            data = [0, 1, 0, 0]
        elif diagnostics == "rsu6":
            data = [0, 0, 1, 0]
        else:
            data = [0, 0, 0, 1]
    if topic == "demo/test/obu":
        print("OBU:")
        obumetrics = received["metrics"]
        timestamp = received["timestamp"]

        for metric in obumetrics:
            sender_host = metric['hostname']
            topic = metric['metric']
            value = metric['value']
            print(sender_host, topic, value)
            writeToDb(sender_host, topic, value, timestamp)
    else:
        print(f"{sender_host}:".upper())
        metrics = received["metrics"]

        for metric in metrics:
            topic = metric['metric']
            value = metric['value']['metric_value']
            timestamp = metric["value"]["timestamp"]
            print(topic, value)
            writeToDb(sender_host, topic, value, timestamp)


# WARNING, you MUST store the return value in order for the subscription to work!!
# This is because if you don't, the reference counter will reach 0 and the subscription
# will be immediately undeclared.
sub = session.declare_subscriber(
    key, listener, reliability=Reliability.RELIABLE())

print("Enter 'q' to quit...")
c = '\0'
while c != 'q':
    c = sys.stdin.read(1)
    if c == '':
        time.sleep(1)

# Cleanup: note that even if you forget it, cleanup will happen automatically when
# the reference counter reaches 0
sub.undeclare()
session.close()
