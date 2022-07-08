# idea:
# class RemoteSequence:
#     """A chain of specific remote peers; created by RemoteSequenceManager.make_sequence()"""
#     spans: Sequence[Span] # spans that describe which specific modules are assigned to which remote
#     # note: RemoteSequenceManager.make_sequence should use load balancing!
#
# def RemoteSequential(nn.Module):
#     def forward(self, inputs: torch.Tensor):
#         return RemoteSequentialAutogradFunction.apply(inputs, self.sequence_manager(), **self.todo_stuff())
#     def inference_sesion(self, **stuff):
#         self.remote_sequence_info.update_()
#         return RemoteSequentialInferenceSession(self.remote_sequence_info, self.p2p)

# class _RemoteSequentialCall(torch.autograd.Function):
#     """
#     A pytorch autograd-compatible function that calls a sequence of transformer blocks on remote peers

#     :note: this function splits input data into batches for efficient parallel processing
#     :note: forward and backward passes may sometimes be served by different modules!
#     """
#
#     def forward(ctx, inputs: torch.Tensor):
#         input_batches: List[torch.Tensor] = split_into_batches(inputs, MAX_TOKENS_PER_BATCH)
#         forward_passes: List[concurrent.futures.Future] = []
#         for input_batch in input_batches:
#             coro = RemoteExpertWorker.run_coroutine(
#               async_forward_pass(RemoteSequenceManager, input_batch)), return_future=True
#             )  # ^-- async_foward_pass does runs RemoteSequenceManager.form_sequence() and runs forward pass in a chain
#             #    if spans[i] breaks, use RemoteSequenceManager[spans[i].start : spans[i].end].form_sequence() to repair
#         output_batches = concurrent.futures.wait(forward_passes)
#         save_intermediate_states(ctx, forward_passes)  # save both your sequence and intermediate states.
#         # ^-- sequence from forward pass is reused for backward! - and repaired the same way
#         # [IMPORTANT] maybe first create an op for one batch, then a wrapper that split into batches
#         return torch.cat(output_batches, dim=0)
#
#    def backward(ctx, *grad_outputs: torch.Tensor):
#         return TODO_think_through(ctx, *grad_outputs)

# class RemoteSequentialInferenceSession:
#    """An interface to a multi-step *inference* session for a sequence of remote transformer blocks"""
#    # form sequence, maintain past inputs for all stages; if any stage breaks, fix the same way as in forward:
#    # patch = RemoteSequenceManager[spans[BROKEN_INDEX].start : spans[BROKEN_INDEX].end].form_sequence()
#    # feed_past_key_values(into=patch, inputs=past_key_values[BROKEN_INDEX])
#    # self.TODO_replace_in_metadata(remove=BROKEN_INDEX, replace_with=patch)
#    #
#    TODO_think()
#
