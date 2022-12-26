import asyncio


async def shield_and_wait(task):
    """
    Works like asyncio.shield(), but waits for the task to finish before raising CancelledError to the caller.
    """

    if not isinstance(task, asyncio.Task):
        task = asyncio.create_task(task)

    cancel_exc = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as e:
            cancel_exc = e
    if cancel_exc is not None:
        raise cancel_exc
    return result
