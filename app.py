import os

import chainlit as cl

from qachatbot.bot.bot import init_settings, setup_chatbot, setup_ragbot
from qachatbot.bot.chat import (
    process_command,
    process_rag,
    process_response,
    process_response_with_vision,
)
from qachatbot.settings import vectorstore_manager


@cl.on_chat_start
async def on_chat_start():
    settings = await init_settings()
    await setup_agent(settings)
    print("A new chat session has started!")


@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")


@cl.on_message
async def on_message(message: cl.Message):
    content = message.content

    response = f"Received: {content}"

    if content.startswith("/"):
        response = process_command(content)
        await cl.Message(response).send()

    else:
        chat_mode = cl.user_session.get("chat_mode")
        if chat_mode == "rag":
            try:
                if message.elements:
                    await process_response_with_vision(message)
                else:
                    response = await process_rag(message.content)
            except Exception as e:
                await cl.Message(response).send()
                raise e


@cl.on_settings_update
async def setup_agent(settings):
    # print("settings:", settings)
    chat_mode = settings["chat_mode"]
    cl.user_session.set("chat_mode", chat_mode)
    # print("DB Mode", cl.user_session.get("DB"))
    match settings["DB"]:
        case "Chroma":
            cl.user_session.set("vectorstore", vectorstore_manager.chromadb)

    match chat_mode:
        case "rag":
            setup_ragbot(settings)
