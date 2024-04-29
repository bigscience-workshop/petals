import torch

PUBLIC_INITIAL_PEERS = [
    # IPv4 DNS addresses
    "/dns/dht1.cillium.dev.compute.agentartificial.com/tcp/8008/p2p/Qmb3skfrki1PR8ww6nxvoGm51F5imK3e1DPMZgtay6ofE2"
]

# The reachability API is currently used only when connecting to the public swarm
REACHABILITY_API_URL = "https://health.cillium.dev.compute.agentartificial.com"

DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")
