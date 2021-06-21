import paramiko
from Crypto.PublicKey import RSA
from Crypto import Random

from config import *


COMMANDS = []
COMMANDS.append('cd SDN/host/ && sudo python3 encrypt.py')


def generate_key():
    random_generator = Random.new().read
    keypair = RSA.generate(Config.KEY_LENGTH, random_generator)
    pub_key = keypair.publickey().exportKey()
    with open (Config.BASE_PATH + 'private.pem', 'wb') as f:
        f.write(keypair.exportKey())
    with open (Config.BASE_PATH + 'public.pem', 'wb') as f:
        f.write(pub_key)


def send_key(host_ips, commands):
    #print("=====generating key=====")
    generate_key()
    #print("=====distributing key=====")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for ip in host_ips:
        ssh.connect(ip, username=Config.HOST_USERNAME, password=Config.HOST_PASSWORD)
        sftp = ssh.open_sftp()
        sftp.put(Config.BASE_PATH + 'public.pem', Config.HOST_BASE_PATH + 'public.pem')
        sftp.close()
        for command in commands:
            stdin, stdout, stderr = ssh.exec_command(command)
            lines = stdout.readlines()
            for line in lines:
                print(line)
            lines = stderr.readlines()
            for line in lines:
                print(line)
        ssh.close()
    #print("=====end program=====")


if __name__ == '__main__':
    send_key(Config.HOST_IPS, COMMANDS)