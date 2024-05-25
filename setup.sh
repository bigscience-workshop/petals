#!/bin/bash

set -e

docker compose --profile miner --env-file envs/dht1.cillium.dev.compute.agentartificial.com.txt up -d 
#sudo docker compose --profile miner --env-file envs/gpu/h100/dht1.cillium.dev.compute.agentartificial.com.txt up -d 
