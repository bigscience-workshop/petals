#!/bin/bash

set -e

if [ -z "$INITIAL_PEERS" ]; then
    INITIAL_PEERS=cat envs/gpu/h100/peers.txt
fi

cat env.example > .env
sed -i "s/INITIAL_PEERS=.*/INITIAL_PEERS=$INITIAL_PEERS/" .env

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install setuptools wheel gnureadline
pip install -e .

docker compose --profile core --env-file .env up -d