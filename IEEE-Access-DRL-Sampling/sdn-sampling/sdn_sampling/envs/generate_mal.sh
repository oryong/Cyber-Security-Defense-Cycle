#!/bin/bash

sshpass -p 2229 ssh root@172.26.17.158 "
echo '2229' | \\
sudo ./SYN_attack.sh;
exit;
"