import requests
import json
import time

from config import *


def get_all_nodes():
    url = Config.BASE_URL + '/switchmanager/default/nodes'
    response = requests.get(url, auth=(Config.ODL_USERNAME, Config.ODL_PASSWORD))
    node_list = response.json()['nodeProperties']
    
    return node_list


def get_node_stats(nodeid):
    req_str = Config.BASE_URL + '/statistics/default/port/node/OF/' + nodeid
    response = requests.get(req_str, auth=(Config.ODL_USERNAME, Config.ODL_PASSWORD))
    node_stats = response.json()
    
    return node_stats


def aggregate_traffic():
    agg = 0
    node_list = get_all_nodes()
    for node in node_list:
        node_stats = get_node_stats(node['node']['id'])
        nodeid = node_stats['node']['id']    
        for stats in node_stats['portStatistic']:
            agg = agg + int(stats['transmitBytes']) + int(stats['receiveBytes'])
    
    return agg


def measure():
    old_agg = 0
    cnt = 0
    while True:
        agg = aggregate_traffic()
        rate = (agg - old_agg)/1000000000.0
        old_agg = agg
        result = str(cnt) + " sec Aggregate Traffic (tx + rx) = " + str(rate) + " GBytes/sec"
        print(result)
        with open('traffic_result2.txt', 'a') as f:
             f.write(result + '\n')
        time.sleep(5)
        cnt += 5


if __name__=='__main__':
    measure()
