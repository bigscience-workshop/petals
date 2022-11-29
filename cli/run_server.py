import argparse

import configargparse
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.limits import increase_file_limit
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from humanfriendly import parse_size

from src.constants import PUBLIC_INITIAL_PEERS
from src.server.server import Server

use_hivemind_log_handler("in_root_logger")
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
    parser.add_argument('--host_maddrs', nargs='+', default=['/ip4/0.0.0.0/tcp/0', '/ip6/::/tcp/0'], required=False,
                        help='Multiaddrs to listen for external connections from other peers. Default: all IPv4/IPv6 interfaces, a random free TCP port')
    parser.add_argument('--announce_maddrs', nargs='+', default=None, required=False,
                        help='Visible multiaddrs the host announces for external connections from other peers')

    parser.add_argument('--compression', type=str, default='NONE', required=False, help='Tensor compression communication')

    parser.add_argument('--num_handlers', type=int, default=8, required=False,
                        help='server will use this many processes to handle incoming requests')
    parser.add_argument('--min_batch_size', type=int, default=1,
                        help='Minimum required batch size for all operations (in total tokens)')
    parser.add_argument('--max_batch_size', type=int, default=16384,
                        help='The total number of tokens in the same batch will not exceed this value')
    parser.add_argument('--prefetch_batches', type=int, default=1, required=False,
                        help='Pre-form this many subsequent batches while GPU is processing the current one')
    parser.add_argument('--sender_threads', type=int, default=1, required=False,
                        help='Use this many threads to pass results/exceptions from Runtime to Pools')
    parser.add_argument('--inference_max_length', type=int, default=16384,
                        help='Maximum total sequence length permitted per inference, defaults to 16384 tokens')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.')
    parser.add_argument('--device', type=str, default=None, required=False,
                        help='all blocks will use this device in torch notation; default: cuda if available else cpu')
    parser.add_argument("--torch_dtype", type=str, default="auto",
                        help="Use this dtype to store block weights and do computations. "
                             "By default, respect the dtypes in the pre-trained state dict.")
    parser.add_argument('--attn_cache_size', type=str, default=None,
                        help='The size of GPU memory allocated for storing past attention keys/values between inference'
                             ' steps; examples: 500MB or 1.2GB or 1073741824 (bytes); be warned: 1KB != 1KiB')
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
    parser.add_argument('--update_period', type=float, required=False, default=30,
                        help='Server will report blocks to DHT once in this many seconds')
    parser.add_argument('--expiration', type=float, required=False, default=None,
                        help='DHT entries will expire after this many seconds')
    parser.add_argument('--request_timeout', type=float, required=False, default=3 * 60,
                        help='Timeout for the whole rpc_forward/rpc_backward/rpc_forward_stream/rpc_backward_stream request')
    parser.add_argument('--session_timeout', type=float, required=False, default=30 * 60,
                        help='Timeout for the whole inference session')
    parser.add_argument('--step_timeout', type=float, required=False, default=5 * 60,
                        help="Timeout for waiting the next step's inputs inside an inference session")

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

    parser.add_argument("--use_auth_token", type=str, default=None, help="auth token for from_pretrained")
    parser.add_argument('--load_in_8bit', action='store_true', help='Convert the loaded model into mixed-8bit quantized model.')

    # fmt:on
    args = vars(parser.parse_args())
    args.pop("config", None)

    args["converted_model_name_or_path"] = args.pop("model") or args["converted_model_name_or_path"]

    if args.pop("increase_file_limit"):
        increase_file_limit()

    compression_type = args.pop("compression").upper()
    compression = getattr(CompressionType, compression_type)

    attn_cache_size = args.pop("attn_cache_size")
    if attn_cache_size is not None:
        attn_cache_size = parse_size(attn_cache_size)
    assert isinstance(
        attn_cache_size, (int, type(None))
    ), "unrecognized value for attention_cache_bytes, examples: 1.5GB or 1500MB or 1572864000 (bytes)"

    if args.pop("new_swarm"):
        args["initial_peers"] = []

    use_auth_token = args.pop("use_auth_token")
    args["use_auth_token"] = True if use_auth_token in ("True", "true", "") else use_auth_token

    server = Server(**args, compression=compression, attn_cache_size=attn_cache_size)
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, shutting down")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
