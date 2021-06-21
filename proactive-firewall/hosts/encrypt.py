import os
from Crypto.PublicKey import RSA

from config import *


def get_ip_name():
    f = os.popen('ifconfig eth1 | grep "inet\ addr" | cut -d: -f2 | cut -d" " -f1')
    ip = f.read().strip()
    if ip == '172.26.17.121':
        ip = '1'
    elif ip == '172.26.17.122':
        ip = '2'
    elif ip == '172.26.17.123':
        ip = '3'
    elif ip == '172.26.17.124':
        ip = '4'
    elif ip == '172.26.17.125':
        ip = '5'
    elif ip == '172.26.17.126':
        ip = '6'
    elif ip == '172.26.17.127':
        ip = '7'
    elif ip == '172.26.17.128':
        ip = '8'
    elif ip == '172.26.17.129':
        ip = '9'
    elif ip == '172.26.17.130':
        ip = '10'
    elif ip == '172.26.17.131':
        ip = '11'
    elif ip == '172.26.17.132':
        ip = '12'

    rules_file = Config.BASE_PATH + 'host_pi_' + ip + '.rules'

    return rules_file


def save_iptables(rules_file):
    os.system('iptables-save > ' + rules_file)


def load_key():
    with open (Config.BASE_PATH + 'public.pem', 'rb') as f:
        pub_key = RSA.importKey(f.read())
    
    return pub_key


def encrypt(rules_file, key):
    enc_file = rules_file + '.enc'
    with open(rules_file, 'r') as f, open(enc_file, 'wb') as enc_f:
        while True:
            data = f.read(Config.CHUNK_SIZE)
            if data == '':
                break
            encrypted = key.encrypt(data.encode(), 32)
            enc_f.write(encrypted[0])
    
    return enc_file


def make_enc_file():
    rules_file = get_ip_name()
    save_iptables(rules_file)
    key = load_key()
    enc_file = encrypt(rules_file, key)


if __name__ == '__main__':
    make_enc_file()