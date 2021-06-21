#!/bin/bash

rate1=${1}
rate2=${2}
rate3=${3}
rate4=${4}
rate5=${5}
rate6=${6}
rate7=${7}
rate8=${8}
rate9=${9}
rate10=${10}
epsilon=${11}

ssh wits_controller@172.26.17.82 "
echo '2229' | \\
sudo -kS python /home/wits_controller/DRL_sampling/drl_sampling_v0.py \\
$rate1 $rate2 $rate3 $rate4 $rate5 $rate6 $rate7 $rate8 $rate9 $rate10 $epsilon;
exit;
"
