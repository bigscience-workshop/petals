#!/bin/bash

set -e

cp envs/dht1.cillium.dev.compute.agentartificial.com.txt  .env
docker compose --profile miner --env-file .env up -d 
