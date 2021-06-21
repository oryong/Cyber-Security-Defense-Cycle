import random
import json

from config import *


def get_rules_files():
    rules_files = []
    #for idx, ip in enumerate(Config.HOST_IPS):
    for idx in range(12):
        if idx != 2:
            rules_file = Config.BASE_PATH + 'controller_pi_' + str(idx+1) + '.rules'
            rules_files.append(rules_file)
    
    return rules_files


def convert(rules_file, flow_id):
    with open(rules_file, 'r') as f:
        rules = f.readlines()
        for rule in rules:
            rule = rule.split()
            is_flow = False
            for sw_idx, switch in enumerate(Config.SWITCHS):
                if '-A' in rule:
                    is_flow = True
                    is_ip_source = False
                    is_ip_dest = False
                    is_port_source = False
                    is_port_dest = False
                    is_udp = False
                    is_tcp = False
                    is_drop = False
                    if '-s' in rule:
                        is_ip_source = True
                        idx = rule.index('-s')
                        ip_source = rule[idx+1]
                    if '-d' in rule:
                        is_ip_dest = True
                        idx = rule.index('-d')
                        ip_dest = rule[idx+1]
                    if '--sport' in rule:
                        is_port_source = True
                        idx = rule.index('--sport')
                        port_source = rule[idx+1]
                    if '--dport' in rule:
                        is_port_dest = True
                        idx = rule.index('--dport')
                        port_dest = rule[idx+1]
                    if '-p' in rule:
                        idx = rule.index('-p')
                        if 'tcp' == rule[idx+1]:
                            is_tcp = True
                        elif 'udp' == rule[idx+1]:
                            is_udp = True
                    if '-j' in rule:
                        idx = rule.index('-j')
                        if 'DROP' == rule[idx+1]:
                            is_drop = True

                    flow_name = 'sw' + str(sw_idx+1) + '_flow' + str(flow_id)
                    new_flow = {}
                    new_flow['name'] = flow_name
                    new_flow['installInHw'] = 'true'
                    new_flow['node'] = {u'id': switch, u'type': u'OF'}
                    new_flow['etherType'] = 0x800
                    if (is_ip_source):
                        new_flow['nwSrc'] = ip_source
                    if (is_ip_dest):
                        new_flow['nwDst'] = ip_dest
                    if (is_tcp):
                        new_flow['protocol'] = 0x6
                    if (is_udp):
                        new_flow['protocol'] = 0x11
                    if (is_port_source):
                        new_flow['tpSrc'] = port_source
                    if (is_port_dest):
                        new_flow['tpSrc'] = port_dest
                    new_flow['priority'] = str(random.randint(1, 500))
                    if (is_drop):
                        new_flow['actions'] = ['DROP']
                    with open(Config.BASE_PATH + flow_name + '.json', 'w') as f:
                        json_file = json.dumps(new_flow)
                        f.write(json_file)
            if is_flow:
                flow_id += 1

    return flow_id


def convert_all():
    rules_files = get_rules_files()
    flow_id = 1
    for rules_file in rules_files:
        flow_id = convert(rules_file, flow_id)


if __name__=="__main__":
    convert_all()