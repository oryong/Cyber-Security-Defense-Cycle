import socket
import paramiko
import time
from Crypto.PublicKey import RSA
from Crypto import Random

from config import *
from encrypt import *


RETRY = []
RETRY.append('cd SDN/host/ && sudo python3 encrypt.py')


def load_key():
    with open (Config.BASE_PATH + 'private.pem', 'rb') as f:
        priv_key = RSA.importKey(f.read())
    
    return priv_key


def ip_convert(ip):
    if ip == '172.26.17.121':
        ip = '1'
    elif ip == '172.26.17.122':
        ip = '2'
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

    return ip


def recv_file(host_ips):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for ip in host_ips:
        ssh.connect(ip, username=Config.HOST_USERNAME, password=Config.HOST_PASSWORD)
        name = ip_convert(ip)
        sftp = ssh.open_sftp()
        sftp.get(Config.HOST_BASE_PATH + "host_pi_" + name + ".rules.enc", Config.BASE_PATH + "controller_pi_" + name + ".rules.enc")
        sftp.close()
        ssh.close()


def dec_files(priv_key, host_ips):
    retry_hosts = []
    for ip in host_ips:
        name = ip_convert(ip)
        enc_file = Config.BASE_PATH + "controller_pi_" + name + ".rules.enc"
        rules_file = enc_file[:-4]
        with open(enc_file, 'rb') as enc_f, open(rules_file, 'w') as dec_f:
            while True:
                encrypted = enc_f.read(Config.CHUNK_SIZE)
                if encrypted == b'':
                    break
                decrypted = priv_key.decrypt(encrypted)
                try:
                    dec_f.write(decrypted.decode())
                except:
                    retry_hosts.append(ip)
                    dec_f.write(b'-error'.decode())
                    break

    return retry_hosts


def retry(retry_hosts):
    send_key(retry_hosts, RETRY)
    priv_key = load_key()
    recv_file(retry_hosts)
    retry_hosts = dec_files(priv_key, retry_hosts)
    if retry_hosts:
        retry(retry_hosts)


def collect_all():
    print("=====loading key=====")
    priv_key = load_key()
    print("=====receiving data=====")
    recv_file(Config.HOST_IPS)
    print("=====decrypting data=====")
    retry_hosts = dec_files(priv_key, Config.HOST_IPS)
    if retry_hosts:
        retry(retry_hosts)
    print("=====end program=====")


if __name__ == '__main__':
    collect_all()