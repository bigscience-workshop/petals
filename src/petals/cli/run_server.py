import argparse

import configargparse
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.limits import increase_file_limit
from hivemind.utils.logging import get_logger
from humanfriendly import parse_size

from petals.constants import PUBLIC_INITIAL_PEERS
from petals.server.server import Server

logger = get_logger(__file__)


def main():
    # fmt:off
    parser = configargparse.ArgParser(default_config_files=["config.yml"],
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--converted_model_name_or_path', type=str, default=None,
                       help="path or name of a pretrained model, converted with cli/convert_model.py")
    group.add_argument('model', nargs='?', type=str, help="same as --converted_model_name_or_path")

    parser.add_argument('--num_blocks', type=int, default=None, help="The number of blocks to serve")
    parser.add_argument('--block_indices', type=str, default=None, help="Specific block indices to serve")
    parser.add_argument('--prefix', type=str, default=None, help="Announce all blocks with this prefix. By default,"
                                                                 "use the same name as in the converted model.")

    parser.add_argument('--port', type=int, required=False,
                        help='Port this server listens to. '
                             'This is a simplified way to set the --host_maddrs and --announce_maddrs options (see below) '
                             'that sets the port across all interfaces (IPv4, IPv6) and protocols (TCP, etc.) '
                             'to the same number. Default: a random free port is chosen for each interface and protocol')
    parser.add_argument('--public_ip', type=str, required=False,
                        help='Your public IPv4 address, which is visible from the Internet. '
                             'This is a simplified way to set the --announce_maddrs option (see below).'
                             'Default: server announces IPv4/IPv6 addresses of your network interfaces')

    parser.add_argument('--host_maddrs', nargs='+', required=False,
                        help='Multiaddrs to listen for external connections from other peers')
    parser.add_argument('--announce_maddrs', nargs='+', required=False,
                        help='Visible multiaddrs the host announces for external connections from other peers')

    parser.add_argument('--compression', type=str, default='NONE', required=False, help='Tensor compression communication')

    parser.add_argument('--num_handlers', type=int, default=8, required=False,
                        help='server will use this many processes to handle incoming requests')
    parser.add_argument('--min_batch_size', type=int, default=1,
                        help='Minimum required batch size for all operations (in total tokens)')
    parser.add_argument('--max_batch_size', type=int, default=2048,
                        help='The total number of tokens in the same batch will not exceed this value')
    parser.add_argument('--prefetch_batches', type=int, default=1, required=False,
                        help='Pre-form this many subsequent batches while GPU is processing the current one')
    parser.add_argument('--sender_threads', type=int, default=1, required=False,
                        help='Use this many threads to pass results/exceptions from Runtime to Pools')
    parser.add_argument('--inference_max_length', type=int, default=2048,
                        help='Maximum total sequence length permitted per inference, defaults to 16384 tokens')

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
    parser.add_argument("--torch_dtype", type=str, default="auto",
                        help="Use this dtype to store block weights and do computations. "
                             "By default, respect the dtypes in the pre-trained state dict.")
    parser.add_argument('--attn_cache_size', type=str, default=None,
                        help='The size of GPU memory allocated for storing past attention keys/values between inference steps. '
                             'Examples: 500MB, 1.2GB, 1073741824 (bytes). Note that 1KB != 1KiB here. '
                             'Default: 0.5GiB * num_blocks * hidden_size / 14336. '
                             'The latter is the hidden size of the bigscience/bloom-petals model.')
    parser.add_argument('--alloc_timeout', type=float, default=60,
                        help='If the cache is full, the server will wait for this number of seconds hoping that some memory will be freed '
                             'before rejecting the request')
    parser.add_argument('--revision', type=str, default='main',
                        help="The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models"
                             "and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.")

    parser.add_argument('--throughput',
                        type=lambda value: value if value in ['auto', 'eval'] else float(value),
                        default='auto',
                        help='Expected server throughput (a float measured in RPS). '
                             'If set to "auto" (default), the script evaluates network and compute throughput '
                             'on the first run and uses these estimates for future runs. '
                             'If set to "eval", the script re-evaluates the throughput and overrides the cache.')
    parser.add_argument('--update_period', type=float, required=False, default=150,
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
    group.add_argument('--initial_peers', type=str, nargs='*', required=False, default=PUBLIC_INITIAL_PEERS,
                       help='Multiaddrs of one or more DHT peers from the target swarm. Default: connects to the public swarm')
    group.add_argument('--new_swarm', action='store_true',
                       help='Start a new private swarm (i.e., do not connect to any initial peers)')

    parser.add_argument('--increase_file_limit', action='store_true',
                        help='On *nix, this will increase the max number of processes '
                             'a server can spawn before hitting "Too many open files"; Use at your own risk.')
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

    parser.add_argument("--use_auth_token", action='store_true', help="auth token for from_pretrained")
    parser.add_argument('--load_in_8bit', type=str, default=None,
                        help="Convert the loaded model into mixed-8bit quantized model. "
                             "Default: True if GPU is available. Use `--load_in_8bit False` to disable this")

    parser.add_argument("--skip_reachability_check", action='store_true',
                        help="Skip checking this server's reachability via health.petals.ml "
                             "when connecting to the public swarm. If you connect to a private swarm, "
                             "the check is skipped by default. Use this option only if you know what you are doing")

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

    if args.pop("increase_file_limit"):
        increase_file_limit()

    compression_type = args.pop("compression").upper()
    compression = getattr(CompressionType, compression_type)

    attn_cache_size = args.pop("attn_cache_size")
    if attn_cache_size is not None:
        attn_cache_size = parse_size(attn_cache_size)
    assert isinstance(
        attn_cache_size, (int, type(None))
    ), "Unrecognized value for --attn_cache_size. Correct examples: 1.5GB or 1500MB or 1572864000 (bytes)"

    max_disk_space = args.pop("max_disk_space")
    if max_disk_space is not None:
        max_disk_space = parse_size(max_disk_space)
    assert isinstance(
        max_disk_space, (int, type(None))
    ), "Unrecognized value for --max_disk_space. Correct examples: 1.5GB or 1500MB or 1572864000 (bytes)"

    if args.pop("new_swarm"):
        args["initial_peers"] = []

    load_in_8bit = args.pop("load_in_8bit")
    if load_in_8bit is not None:
        args["load_in_8bit"] = load_in_8bit.lower() in ["true", "1"]

    server = Server(
        **args,
        host_maddrs=host_maddrs,
        announce_maddrs=announce_maddrs,
        compression=compression,
        max_disk_space=max_disk_space,
        attn_cache_size=attn_cache_size,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, shutting down")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
