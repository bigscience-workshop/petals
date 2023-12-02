import torch
from torch.utils._pytree import tree_flatten as _tree_flatten, tree_unflatten as _tree_unflatten


def make_inference_graphed_callable(callable: callable, sample_args, num_warmup_iters=3):
    """Similar to torch.cuda.make_graphed_callables, but takes only one function and does not build a graph for the backward pass"""
    assert not isinstance(callable, torch.nn.Module)
    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise RuntimeError(
            "make_graphed_callables does not support the autocast caching. Please set `cache_enabled=False`."
        )

    flatten_arg, _ = _tree_flatten(sample_args)
    flatten_sample_args = tuple(flatten_arg)
    assert all(
        isinstance(arg, torch.Tensor) for arg in flatten_arg
    ), "In the beta API, sample_args for each callable must contain only Tensors. Other types are not allowed."

    len_user_args = len(sample_args)
    static_input_surface = flatten_sample_args

    graph = torch.cuda.CUDAGraph()

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(num_warmup_iters):
            outputs, _ = _tree_flatten(callable(*sample_args))
        del outputs
    torch.cuda.current_stream().wait_stream(s)

    # Capture forward graph
    with torch.cuda.graph(graph):
        outputs = callable(*sample_args)

    flatten_outputs, output_unflatten_spec = _tree_flatten(outputs)
    static_outputs = tuple(flatten_outputs)

    def make_graphed_function(
        graph,
        len_user_args,
        output_unflatten_spec,
        static_input_surface,
        static_outputs,
    ):
        def replay_graph(*inputs):
            # At this stage, only the user args may (potentially) be new tensors.
            for i in range(len_user_args):
                if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                    static_input_surface[i].copy_(inputs[i])
            graph.replay()
            assert isinstance(static_outputs, tuple)
            return tuple(o.detach() for o in static_outputs)

        def functionalized(*user_args):
            # Runs the autograd function with inputs == all inputs to the graph that might require grad
            # (explicit user args + module parameters)
            # Assumes module params didn't change since capture.
            flatten_user_args, _ = _tree_flatten(user_args)
            out = replay_graph(*flatten_user_args)
            return _tree_unflatten(out, output_unflatten_spec)

        return functionalized

    # Put together the final graphed callable
    graphed = make_graphed_function(
        graph,
        len_user_args,
        output_unflatten_spec,
        static_input_surface,
        static_outputs,
    )
    return graphed
