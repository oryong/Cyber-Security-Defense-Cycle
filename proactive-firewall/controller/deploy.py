import os
import requests
import json

from config import *


def deploy_all():
    print("=====deploying rules=====")
    for i in range(1, 3301):
        for idx, switch in enumerate(Config.SWITCHS):
            flow_name = 'sw' + str(idx+1) + '_flow' + str(i)
            flow_file = Config.BASE_PATH + flow_name + '.json'
            if os.path.isfile(flow_file):
                URL = Config.BASE_URL + '/flowprogrammer/default/node/OF/' + switch + '/staticFlow/' + flow_name
                headers = {'Content-Type': 'application/json'}
                data = open(flow_file, 'r').read()
                response = requests.put(URL, headers=headers, data=data, auth=(Config.ODL_USERNAME, Config.ODL_PASSWORD))
                if response.content != b'Success':
                    print(response.content)
            else:
                break
    print("=====end program=====")


if __name__=='__main__':
    deploy_all()