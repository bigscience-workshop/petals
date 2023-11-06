# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py

import argparse
import asyncio
import dataclasses
import json
import time
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from packaging import version
from hivemind import get_logger
from transformers import GenerationConfig, AutoTokenizer

from petals import AutoDistributedModelForCausalLM
from petals.api.openai.protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    LogProbs, ModelCard, ModelList, ModelPermission, UsageInfo, random_uuid
)

from petals.data_structures import FromPretrainedInfo

try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template

    _fastchat_available = True
except ImportError:
    _fastchat_available = False

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = get_logger(__name__)

app = fastapi.FastAPI()
model = None
tokenizer = None


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error").dict(),
                        status_code=status_code.value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


async def get_gen_prompt(request) -> str:
    if not _fastchat_available:
        raise ModuleNotFoundError(
            "fastchat is not installed. Please install fastchat to use "
            "the chat completion and conversation APIs: `$ pip install fschat`"
        )
    if version.parse(fastchat.__version__) < version.parse("0.2.23"):
        raise ImportError(
            f"fastchat version is low. Current version: {fastchat.__version__} "
            "Please upgrade fastchat to use: `$ pip install -U fschat`")

    conv = get_conversation_template(request.model)
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system_message = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    return prompt


async def check_length(
        request: Union[ChatCompletionRequest, CompletionRequest],
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None
) -> Tuple[List[int], Optional[JSONResponse]]:
    assert (not (prompt is None and prompt_ids is None)
            and not (prompt is not None and prompt_ids is not None)
            ), "Either prompt or prompt_ids should be provided."
    if prompt_ids is not None:
        input_ids = prompt_ids
    else:
        input_ids = tokenizer(prompt).input_ids
    token_num = len(input_ids)

    if request.max_tokens is None:
        request.max_tokens = max_model_len - token_num
    if token_num + request.max_tokens > max_model_len:
        return input_ids, create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return input_ids, None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=served_model,
                  root=served_model,
                  permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)


def create_logprobs(token_ids: List[int],
                    id_logprobs: List[Dict[int, float]],
                    initial_text_offset: int = 0) -> LogProbs:
    """Create OpenAI-style logprobs."""
    logprobs = LogProbs()
    last_token_len = 0
    for token_id, id_logprob in zip(token_ids, id_logprobs):
        token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(id_logprob[token_id])
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] +
                                        last_token_len)
        last_token_len = len(token)

        logprobs.top_logprobs.append({
            tokenizer.convert_ids_to_tokens(i): p
            for i, p in id_logprob.items()
        })
    return logprobs


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by Petals engine)
    """
    logger.info(f"Received chat completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in Petals engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    prompt = await get_gen_prompt(request)
    token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    try:
        skip_special_tokens = request.skip_special_tokens
        spaces_between_special_tokens = request.spaces_between_special_tokens

        if request.ignore_eos:
            logger.warning("ignore_eos is not supported yet")

        if request.presence_penalty > 0.0:
            logger.warning("presence_penalty is not supported yet")

        generation_config = GenerationConfig(
            num_return_sequences=request.n,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            max_length=request.max_tokens,
            do_sample=True if request.temperature is not None or request.top_p is not None or request.top_k is not None else False,
            num_beams=request.best_of if request.use_beam_search else 1,
            early_stopping=request.stop,
            eos_token_id=request.stop_token_ids,
            repetition_penalty=1.0 + request.presence_penalty if request.presence_penalty else 1.0,
            # output_scores=True,  # Example: set to True if you want to output prediction scores
            return_dict_in_generate=True,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    results_generator = model.generate(token_ids, **generation_config.to_diff_dict())

    # def create_stream_response_json(
    #         index: int,
    #         text: str,
    #         finish_reason: Optional[str] = None,
    # ) -> str:
    #     choice_data = ChatCompletionResponseStreamChoice(
    #         index=index,
    #         delta=DeltaMessage(content=text),
    #         finish_reason=finish_reason,
    #     )
    #     response = ChatCompletionStreamResponse(
    #         id=request_id,
    #         created=created_time,
    #         model=model_name,
    #         choices=[choice_data],
    #     )
    #     response_json = response.json(ensure_ascii=False)
    #
    #     return response_json
    #
    # async def completion_stream_generator() -> AsyncGenerator[str, None]:
    #     # First chunk with role
    #     for i in range(request.n):
    #         choice_data = ChatCompletionResponseStreamChoice(
    #             index=i,
    #             delta=DeltaMessage(role="assistant"),
    #             finish_reason=None,
    #         )
    #         chunk = ChatCompletionStreamResponse(id=request_id,
    #                                              choices=[choice_data],
    #                                              model=model_name)
    #         data = chunk.json(exclude_unset=True, ensure_ascii=False)
    #         yield f"data: {data}\n\n"
    #
    #     previous_texts = [""] * request.n
    #     previous_num_tokens = [0] * request.n
    #     async for res in result_generator:
    #         res: RequestOutput
    #         for output in res.outputs:
    #             i = output.index
    #             delta_text = output.text[len(previous_texts[i]):]
    #             previous_texts[i] = output.text
    #             previous_num_tokens[i] = len(output.token_ids)
    #             response_json = create_stream_response_json(
    #                 index=i,
    #                 text=delta_text,
    #             )
    #             yield f"data: {response_json}\n\n"
    #             if output.finish_reason is not None:
    #                 response_json = create_stream_response_json(
    #                     index=i,
    #                     text="",
    #                     finish_reason=output.finish_reason,
    #                 )
    #                 yield f"data: {response_json}\n\n"
    #     yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        logger.warning("Streaming is not supported yet")
        # return StreamingResponse(completion_stream_generator(),
        #                          media_type="text/event-stream")

    # Non-streaming response
    # async for res in result_generator:
    #     if await raw_request.is_disconnected():
    #         # Abort the request if the client disconnects.
    #         # TODO: Abort the request if the client disconnects.
    #         return create_error_response(HTTPStatus.BAD_REQUEST,
    #                                      "Client disconnected")
    #     final_res = res

    choices = []
    for index, output_tokens in enumerate(results_generator):

        text = tokenizer.decode(
            output_tokens,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

        choice_data = ChatCompletionResponseChoice(
            index=index,
            message=ChatMessage(role="assistant", content=text),
            finish_reason=None,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(token_ids)
    num_generated_tokens = sum(
        len(output) for output in results_generator)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following features:
        - echo (since the Petals engine does not currently support
          getting the logprobs of prompt tokens)
        - suffix (the language models we currently support do not support
          suffix)
        - logit_bias (to be supported by Petals engine)
        - stream (to be supported by Petals engine)
        - logprobs (to be supported by Petals engine)
    """
    logger.info(f"Received completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.echo:
        # We do not support echo since the Petals engine does not
        # currently support getting the logprobs of prompt tokens.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "echo is not currently supported")

    if request.suffix is not None:
        # The language models we currently support do not support suffix.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "suffix is not currently supported")

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in Petals.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"

    use_token_ids = False
    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "please provide at least one prompt")
        first_element = request.prompt[0]
        if isinstance(first_element, int):
            use_token_ids = True
            prompt = request.prompt
        elif isinstance(first_element, (str, list)):
            # TODO: handles multiple prompt case in list[list[int]]
            if len(request.prompt) > 1:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    "multiple prompts in a batch is not currently supported")
            use_token_ids = not isinstance(first_element, str)
            prompt = request.prompt[0]
    else:
        prompt = request.prompt

    if use_token_ids:
        token_ids, error_check_ret = await check_length(request, prompt_ids=prompt)
    else:
        token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    created_time = int(time.monotonic())
    try:
        skip_special_tokens = request.skip_special_tokens
        spaces_between_special_tokens = request.spaces_between_special_tokens

        if request.ignore_eos:
            logger.warning("ignore_eos is not supported yet")

        if request.presence_penalty > 0.0:
            logger.warning("presence_penalty is not supported yet")

        generation_config = GenerationConfig(
            num_return_sequences=request.n,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            max_length=request.max_tokens,
            do_sample=True if request.temperature is not None or request.top_p is not None or request.top_k is not None else False,
            num_beams=request.best_of if request.use_beam_search else 1,
            early_stopping=request.stop,
            eos_token_id=request.stop_token_ids,
            repetition_penalty=1.0 + request.presence_penalty if request.presence_penalty else 1.0,
            # output_scores=True,  # Example: set to True if you want to output prediction scores
            return_dict_in_generate=True,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    results_generator = model.generate(token_ids, **generation_config.to_diff_dict())

    if request.stream:
        # handle streaming case
        logger.warning("Streaming is not supported yet")

    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. In addition, we do not stream the results when use beam search.
    # stream = (request.stream
    #           and (request.best_of is None or request.n == request.best_of)
    #           and not request.use_beam_search)
    #
    # def create_stream_response_json(
    #         index: int,
    #         text: str,
    #         logprobs: Optional[LogProbs] = None,
    #         finish_reason: Optional[str] = None,
    # ) -> str:
    #     choice_data = CompletionResponseStreamChoice(
    #         index=index,
    #         text=text,
    #         logprobs=logprobs,
    #         finish_reason=finish_reason,
    #     )
    #     response = CompletionStreamResponse(
    #         id=request_id,
    #         created=created_time,
    #         model=model_name,
    #         choices=[choice_data],
    #     )
    #     response_json = response.json(ensure_ascii=False)
    #
    #     return response_json
    #
    # async def completion_stream_generator() -> AsyncGenerator[str, None]:
    #     previous_texts = [""] * request.n
    #     previous_num_tokens = [0] * request.n
    #     async for res in result_generator:
    #         res: RequestOutput
    #         for output in res.outputs:
    #             i = output.index
    #             delta_text = output.text[len(previous_texts[i]):]
    #             if request.logprobs is not None:
    #                 logprobs = create_logprobs(
    #                     output.token_ids[previous_num_tokens[i]:],
    #                     output.logprobs[previous_num_tokens[i]:],
    #                     len(previous_texts[i]))
    #             else:
    #                 logprobs = None
    #             previous_texts[i] = output.text
    #             previous_num_tokens[i] = len(output.token_ids)
    #             response_json = create_stream_response_json(
    #                 index=i,
    #                 text=delta_text,
    #                 logprobs=logprobs,
    #             )
    #             yield f"data: {response_json}\n\n"
    #             if output.finish_reason is not None:
    #                 logprobs = (LogProbs()
    #                             if request.logprobs is not None else None)
    #                 response_json = create_stream_response_json(
    #                     index=i,
    #                     text="",
    #                     logprobs=logprobs,
    #                     finish_reason=output.finish_reason,
    #                 )
    #                 yield f"data: {response_json}\n\n"
    #     yield "data: [DONE]\n\n"
    #
    # # Streaming response
    # if stream:
    #     return StreamingResponse(completion_stream_generator(),
    #                              media_type="text/event-stream")

    # Non-streaming response

    # async for res in result_generator:
    #     if await raw_request.is_disconnected():
    #         # TODO: Abort the request if the client disconnects.
    #         return create_error_response(HTTPStatus.BAD_REQUEST,
    #                                      "Client disconnected")
    #     final_res = res

    choices = []

    for index, output_tokens in enumerate(results_generator):
        if request.logprobs is not None:
            # logprobs = create_logprobs(output.token_ids, output.logprobs)
            logger.warning("logprobs is not supported yet")

        text = tokenizer.decode(
            output_tokens,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

        logprobs = None
        choice_data = CompletionResponseChoice(
            index=index,
            text=text,
            logprobs=logprobs,
            finish_reason=None,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(token_ids)
    num_generated_tokens = sum(
        len(output) for output in results_generator)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Petals OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")

    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                             "specified, the model name will be the same as "
                             "the huggingface name.")

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--low_cpu_mem_usage", type=bool, default=True)
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument("--args", type=tuple, default=tuple())

    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    pretrained_info = FromPretrainedInfo(
        model_name_or_path=served_model,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        torch_dtype=args.torch_dtype,
    )

    model = AutoDistributedModelForCausalLM.from_pretrained(*args.args, **dataclasses.asdict(pretrained_info))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, add_bos_token=False)

    max_model_len = model.config.max_position_embeddings

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )
