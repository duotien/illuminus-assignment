import os
from typing import Any, Dict

import chainlit as cl
import PIL.Image
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

from qachatbot.bot.vision import convert_to_base64
from qachatbot.commands import commands


def process_command(content: str):
    content = content.strip()
    cmd = content.split()
    response = f"Unknown command: {cmd[0]}"
    if cmd[0] == "/tp":
        if len(cmd) != 5:
            response = "Wrong syntax!"
        else:
            response = commands.tp(cmd[1], cmd[2], cmd[3], cmd[4])
    return response


async def process_response(message: cl.Message):
    runnable_with_history = cl.user_session.get(
        "runnable_with_history"
    )  # type: Runnable
    response = cl.Message(content="")

    async for chunk in runnable_with_history.astream(
        {
            "question": message.content,
        },
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler()],
            configurable={"session_id": cl.user_session.get("id")},
        ),
    ):
        await response.stream_token(chunk)
    await response.send()
    return response.content


async def process_response_with_vision(message: cl.Message):
    llm_has_clip = cl.user_session.get("llm_has_clip")
    response = cl.Message(content="")
    if not llm_has_clip:
        response.content = "Sorry, it seems like you are trying to send me an image, but I cannot see it."
        await response.send()
        return response
    await process_uploaded(message)


# TODO: add button to change k
async def process_rag(user_input: str, k=5):
    runnable_with_history = cl.user_session.get("runnable_with_history")
    response = cl.Message(content="")
    async for chunk in runnable_with_history.astream(
        user_input,
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler()],
            configurable={"session_id": cl.user_session.get("id")},
        ),
    ):
        await response.stream_token(chunk)
    await response.send()
    return response


# currently limited to 1 file
async def process_uploaded(message):
    for element in message.elements:
        if type(element) == cl.File:
            print("[DEBUG] You uploaded a File")
            await cl.Message(
                content="You uploaded a file, but I cannot process it yet."
            ).send()

        if type(element) == cl.Image:
            print("[DEBUG] You uploaded an Image")
            image = PIL.Image.open(element.path)
            image = image.convert("RGB")
            image = convert_to_base64(image)

            runnable_with_history = cl.user_session.get(
                "runnable_vision_with_history"
            )  # type: Runnable
            response = cl.Message(content="")
            async for chunk in runnable_with_history.astream(
                input={
                    "question": message.content,
                    "image": image,
                },
                config=RunnableConfig(
                    callbacks=[cl.LangchainCallbackHandler()],
                    configurable={"session_id": cl.user_session.get("id")},
                ),
            ):
                await response.stream_token(chunk)
            await response.send()
    return response.content
