import argparse
import logging

import configargparse
import torch
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils import limits
from hivemind.utils.logging import get_logger
from humanfriendly import parse_size

from petals.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS
from petals.server.server import Server
from petals.utils.convert_block import QuantType
from petals.utils.version import validate_version

logger = get_logger(__name__)


def main():
    # fmt:off
    parser = configargparse.ArgParser(default_config_files=["config.yml"],
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--converted_model_name_or_path', type=str, default=None,
                       help="path or name of a pretrained model, converted with cli/convert_model.py")
    group.add_argument('model', nargs='?', type=str, help="same as --converted_model_name_or_path")

    parser.add_argument("--public_name", type=str, default=None, help="Public name to be reported in the leaderboard")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--token", type=str, default=None, help="Hugging Face hub auth token for .from_pretrained()")
    group.add_argument("--use_auth_token", action="store_true", dest="token",
                       help="Read token saved by `huggingface-cli login")

    parser.add_argument('--num_blocks', type=int, default=None, help="The number of blocks to serve")
    parser.add_argument('--block_indices', type=str, default=None, help="Specific block indices to serve")
    parser.add_argument('--dht_prefix', type=str, default=None, help="Announce all blocks with this DHT prefix")

    parser.add_argument('--port', type=int, required=False,
                        help='Port this server listens to. '
                             'This is a simplified way to set the --host_maddrs and --announce_maddrs options (see below) '
                             'that sets the port across all interfaces (IPv4, IPv6) and protocols (TCP, etc.) '
                             'to the same number. Default: a random free port is chosen for each interface and protocol')
    parser.add_argument('--public_ip', type=str, required=False,
                        help='Your public IPv4 address, which is visible from the Internet. '
                             'This is a simplified way to set the --announce_maddrs option (see below).'
                             'Default: server announces IPv4/IPv6 addresses of your network interfaces')

    parser.add_argument("--no_auto_relay", action="store_false", dest="use_auto_relay",
                        help="Do not look for libp2p relays to become reachable if we are behind NAT/firewall")

    parser.add_argument('--host_maddrs', nargs='+', required=False,
                        help='Multiaddrs to listen for external connections from other peers')
    parser.add_argument('--announce_maddrs', nargs='+', required=False,
                        help='Visible multiaddrs the host announces for external connections from other peers')

    parser.add_argument('--daemon_startup_timeout', type=float, default=60,
                        help='Timeout for the libp2p daemon connecting to initial peers')

    parser.add_argument('--compression', type=str, default='NONE', required=False, help='Tensor compression communication')

    parser.add_argument('--num_handlers', type=int, default=8, required=False,
                        help='server will use this many processes to handle incoming requests')
    parser.add_argument('--prefetch_batches', type=int, default=1, required=False,
                        help='Pre-form this many subsequent batches while GPU is processing the current one')
    parser.add_argument('--sender_threads', type=int, default=1, required=False,
                        help='Use this many threads to pass results/exceptions from Runtime to Pools')

    parser.add_argument('--inference_max_length', type=int, default=None,
                        help='Maximum total sequence length permitted per inference, defaults to 16384 tokens. '
                             'Default: 2048 for most models, 8192 for models with multi-query attention (e.g., Llama-2-70b)')
    parser.add_argument('--min_batch_size', type=int, default=1,
                        help='Minimum required batch size for all operations (in total tokens)')
    parser.add_argument('--max_batch_size', type=int, default=None,
                        help='The total number of tokens in the same batch will not exceed this value. '
                             'Default: 2048 for most models, 8192 for models with multi-query attention (e.g., Llama-2-70b)')
    parser.add_argument('--max_chunk_size_bytes', type=int, default=256 * 1024 * 1024,
                        help='Maximum size of activation tensor processed in one go; larger tensors are split into chunks')
    parser.add_argument('--attn_cache_tokens', type=int, default=None,
                        help='The number of past attention key/value pairs that will be stored between inference steps. '
                             'Default: 8192 for most models, 32768 for models with multi-query attention (e.g., Llama-2-70b)')

    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.')
    parser.add_argument("--max_disk_space", type=str, default=None,
                        help="Maximal disk space used for caches. Example: 50GB, 100GiB (GB != GiB here). "
                             "Default: unlimited. "
                             "For bigscience/bloom-petals, this default means that the server may use up to "
                             "min(free_disk_space, 350GB) in the worst case, which happens when the server runs "
                             "for a long time and caches all model blocks after a number of rebalancings. "
                             "However, this worst case is unlikely, expect the server to consume "
                             "the disk space equal to 2-4x of your GPU memory on average.")

    parser.add_argument('--device', type=str, default=None, required=False,
                        help='all blocks will use this device in torch notation; default: cuda if available else cpu')
    parser.add_argument("--torch_dtype", type=str, choices=DTYPE_MAP.keys(), default="auto",
                        help="Use this dtype to store block weights and do computations. "
                             "By default, respect the dtypes in the pre-trained state dict.")
    parser.add_argument('--max_alloc_timeout', type=float, default=600,
                        help="If the cache is full, the server will wait for memory to be freed up to this many seconds"
                             " before rejecting the request")
    parser.add_argument('--revision', type=str, default=None,
                        help="The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models"
                             "and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.")

    parser.add_argument('--throughput',
                        type=lambda value: value if value in ['auto', 'eval', 'dry_run'] else float(value),
                        default='auto',
                        help='Expected server throughput (a float measured in RPS). '
                             'If set to "auto" (default), the script evaluates network and compute throughput '
                             'on the first run and uses these estimates for future runs. '
                             'If set to "eval", the script re-evaluates the throughput and overrides the cache. '
                             'If set to "dry_run", the script re-evaluates the throughput and exits.')
    parser.add_argument('--update_period', type=float, required=False, default=120,
                        help='Server will report blocks to DHT once in this many seconds')
    parser.add_argument('--expiration', type=float, required=False, default=None,
                        help='DHT entries will expire after this many seconds')
    parser.add_argument('--request_timeout', type=float, required=False, default=3 * 60,
                        help='Timeout (in seconds) for the whole rpc_forward/rpc_backward/rpc_forward_stream/rpc_backward_stream request')
    parser.add_argument('--session_timeout', type=float, required=False, default=30 * 60,
                        help='Timeout (in seconds) for the whole inference session')
    parser.add_argument('--step_timeout', type=float, required=False, default=5 * 60,
                        help="Timeout (in seconds) for waiting the next step's inputs inside an inference session")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--initial_peers', type=str, nargs='+', required=False, default=PUBLIC_INITIAL_PEERS,
                       help='Multiaddrs of one or more DHT peers from the target swarm. Default: connects to the public swarm')
    group.add_argument('--new_swarm', action='store_true',
                       help='Start a new private swarm (i.e., do not connect to any initial peers)')

    parser.add_argument('--increase_file_limit', type=int, default=4096,
                        help='On *nix, increase the max number of files a server can open '
                             'before hitting "Too many open files" (set to zero to keep the system limit)')
    parser.add_argument('--stats_report_interval', type=int, required=False,
                        help='Interval between two reports of batch processing performance statistics')

    parser.add_argument('--custom_module_path', type=str, required=False,
                        help='Path of a file with custom nn.modules, wrapped into special decorator')
    parser.add_argument('--identity_path', type=str, required=False, help='Path to identity file to be used in P2P')

    parser.add_argument("--balance_quality", type=float, default=0.75,
                        help="Rebalance the swarm if its throughput is worse than this share of the optimal "
                             "throughput. Use 0.0 to disable rebalancing, values > 1.0 to force rebalancing "
                             "on each check for debugging purposes.")
    parser.add_argument("--mean_balance_check_period", type=float, default=60,
                        help="Check the swarm's balance every N seconds (and rebalance it if necessary)")

    parser.add_argument('--quant_type', type=str, default=None, choices=[choice.name.lower() for choice in QuantType],
                        help="Quantize blocks to 8-bit (int8 from the LLM.int8() paper) or "
                             "4-bit (nf4 from the QLoRA paper) formats to save GPU memory. "
                             "Default: 'int8' if GPU is available, 'none' otherwise")
    parser.add_argument("--tensor_parallel_devices", nargs='+', default=None,
                        help=
                        "Split each block between the specified GPUs such that each device holds a portion of every "
                        "weight matrix. See https://huggingface.co/transformers/v4.9.0/parallelism.html#tensor-parallelism")

    parser.add_argument("--skip_reachability_check", action='store_true',
                        help="Skip checking this server's reachability via health.petals.dev "
                             "when connecting to the public swarm. If you connect to a private swarm, "
                             "the check is skipped by default. Use this option only if you know what you are doing")

    parser.add_argument("--adapters", nargs='*', default=(),
                        help="List of pre-loaded LoRA adapters that can be used for inference or training")

    # fmt:on
    args = vars(parser.parse_args())
    args.pop("config", None)

    args["converted_model_name_or_path"] = args.pop("model") or args["converted_model_name_or_path"]

    host_maddrs = args.pop("host_maddrs")
    port = args.pop("port")
    if port is not None:
        assert host_maddrs is None, "You can't use --port and --host_maddrs at the same time"
    else:
        port = 0
    if host_maddrs is None:
        host_maddrs = [f"/ip4/0.0.0.0/tcp/{port}", f"/ip6/::/tcp/{port}"]

    announce_maddrs = args.pop("announce_maddrs")
    public_ip = args.pop("public_ip")
    if public_ip is not None:
        assert announce_maddrs is None, "You can't use --public_ip and --announce_maddrs at the same time"
        assert port != 0, "Please specify a fixed non-zero --port when you use --public_ip (e.g., --port 31337)"
        announce_maddrs = [f"/ip4/{public_ip}/tcp/{port}"]

    args["startup_timeout"] = args.pop("daemon_startup_timeout")

    file_limit = args.pop("increase_file_limit")
    if file_limit:
        limits.logger.setLevel(logging.WARNING)
        limits.increase_file_limit(file_limit, file_limit)

    compression_type = args.pop("compression").upper()
    compression = getattr(CompressionType, compression_type)

    max_disk_space = args.pop("max_disk_space")
    if max_disk_space is not None:
        max_disk_space = parse_size(max_disk_space)
    assert isinstance(
        max_disk_space, (int, type(None))
    ), "Unrecognized value for --max_disk_space. Correct examples: 1.5GB or 1500MB or 1572864000 (bytes)"

    if args.pop("new_swarm"):
        args["initial_peers"] = []

    quant_type = args.pop("quant_type")
    if quant_type is not None:
        args["quant_type"] = QuantType[quant_type.upper()]

    validate_version()

    if not torch.backends.openmp.is_available():
        # Necessary to prevent the server from freezing after forks
        torch.set_num_threads(1)

    server = Server(
        **args,
        host_maddrs=host_maddrs,
        announce_maddrs=announce_maddrs,
        compression=compression,
        max_disk_space=max_disk_space,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, shutting down")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
