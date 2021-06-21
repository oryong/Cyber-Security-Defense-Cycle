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
mtd_prob=${12}
attacked_group0=${13}
attacked_group1=${14}
attacked_group2=${15}
attacked_group3=${16}

ssh wits_controller@172.26.17.82 "
echo '2229' | \\
sudo -kS python /home/wits_controller/DRL_sampling/DIVERGENCE_v1.py \\
$rate1 $rate2 $rate3 $rate4 $rate5 $rate6 $rate7 $rate8 $rate9 $rate10 $epsilon $mtd_prob $attacked_group0 $attacked_group1 $attacked_group2 $attacked_group3;
exit;
"
