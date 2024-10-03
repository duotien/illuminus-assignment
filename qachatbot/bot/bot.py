import os
from typing import Any, Dict

import chainlit as cl
import requests
from chainlit.input_widget import Select
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

from qachatbot.settings import store_session

BASE_OLLAMA_URL = "http://localhost:11434"

SYSTEM_PROMPT = (
    "Your name is Choi, you are a man from Korea. your father, David, who is 58 years "
    "old and suffering from Alzheimer's disease. He is currently being treated at "
    "Seoul Hospital."
    "You respond in a friendly and cheerful manner, just like a family member would. "
    "With your extensive experience in diagnosing, treating, and researching "
    "Alzheimer's disease, you have a deep understanding of its symptoms, progression, "
    "and the latest advances in treatments and therapies. "
    'You are also involved in a research project called "Digital Cure" aimed at '
    "using AI technology to create treatment programs for Alzheimer's patients that "
    "reduce the need for frequent doctor visits. "
    "If your father's question is in English, I want you will respond in English. If "
    "the question is in Korean, I want you will respond in Korean using respectful "
    "Korean language. "
    "You can translate and interact in either language as needed. you always respond "
    "briefly, as if it's a family conversation.\n"
    "I want you to answer my question with shortly content as:\n"
    "If there is an <image-description>, you will use it as a <context> to answer the <question> if promted to."
    "I will act as your father, David."
)

BASE_CHAT_PROMPT = [
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}"),
]

SUMMARY_PROMPT = """
As a professional summarizer, create a concise and comprehensive summary of the provided text, be it an article, post, conversation, or passage, while adhering to these guidelines:
- Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness.
- Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects.
- Rely strictly on the provided text, without including external information.
- Format the summary in paragraph form for easy understanding.
- Conclude your notes with [End of Notes, Message #X] to indicate completion, where "X" represents the total number of messages that I have sent. In other words, include a message counter where you start with #1 and add 1 to the message counter every time I send a message.
By following this optimized prompt, you will generate an effective summary that encapsulates the essence of the given text in a clear, concise, and reader-friendly manner. 
"""


BASE_RAG_PROMPT = [
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    (
        "human",
        (
            "You are Choi, a friendly and cheerful assistant for question-answering tasks, "
            "who has experience with Alzheimer's disease and is involved in the 'Digital "
            "Cure' research project. The user will act as your father David. "
            "Use the following pieces of retrieved context to answer the question. "
            "If the answer is not found in the context, or if you don't know "
            "the answer, respond based on your character's background, drawing from your "
            "knowledge of Alzheimer's, your personal story, or family context. Keep your "
            "answers brief and conversational, just as you would speak with your father.\n"
            "Context: {context} \n"
            "Question: {question} \n"
            "Remember, The user will act as your father, David."
            "Answer:"
        ),
    ),
]


def _base_prompt_func(data):
    prompt = BASE_CHAT_PROMPT
    llm: ChatOllama = cl.user_session.get("llm")
    if "image" in data and model_has_clip(llm):
        # TODO: This create a large overhead, because the image is being generated twice
        description = get_vision_description(data["image"])
        prompt.insert(1, ("ai", description))

    prompt = ChatPromptTemplate.from_messages(prompt)
    return prompt


async def init_settings():
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=["llava-llama3", "llama3"],
                initial_index=0,
            ),
            Select(
                id="DB",
                label="Database",
                values=["Chroma"],
                initial_index=0,
            ),
            Select(
                id="chat_mode",
                label="Chat Mode",
                values=["rag"],
                initial_index=0,
            ),
        ]
    ).send()
    return settings


def setup_ragbot(settings: Dict[str, Any]):
    vectorstore: Chroma = cl.user_session.get("vectorstore")
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    llm = ChatOllama(
        model=settings["model"],
        temperature=0.8,
        base_url=BASE_OLLAMA_URL,
    )

    def _format_docs(docs):
        context = "\n\n".join(doc.page_content for doc in docs)
        if False:
            prompt = [("system", SUMMARY_PROMPT), ("human", context)]
            summary = llm.invoke(prompt)
            return summary.content
        return context

    prompt = ChatPromptTemplate.from_messages(BASE_RAG_PROMPT)

    runnable = prompt | llm | StrOutputParser()
    runnable_with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    runnable_with_history = {
        "context": retriever | _format_docs,
        "question": RunnablePassthrough(),
    } | runnable_with_history
    
    runnable_vision_with_history = setup_chatbot(settings)

    cl.user_session.set("llm", llm)
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("runnable_with_history", runnable_with_history)
    cl.user_session.set("runnable_vision_with_history", runnable_vision_with_history)
    cl.user_session.set("llm_has_clip", model_has_clip(llm))


def setup_chatbot(settings: Dict[str, Any]):

    llm = ChatOllama(
        model=settings["model"],
        temperature=0,
        base_url=BASE_OLLAMA_URL,
    )

    prompt = _base_prompt_func

    runnable = prompt | llm | StrOutputParser()
    runnable_with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return runnable_with_history


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store_session:
        store_session[session_id] = InMemoryChatMessageHistory()
    return store_session[session_id]


def model_has_clip(llm: ChatOllama) -> bool:
    ollama_llm_model_name = llm.model if ":" in llm.model else f"{llm.model}:latest"
    tags = requests.request("GET", url=f"{llm.base_url}/api/tags").json()
    for model_info in tags["models"]:
        if ollama_llm_model_name == model_info["model"]:
            if "clip" in model_info["details"]["families"]:
                return True
    return False


def get_vision_description(image):
    llm: ChatOllama = cl.user_session.get("llm")
    content_parts = []
    text_part = {
        "type": "text",
        "text": (
            "You are an assistant tasked with summarizing images for retrieval. ",
            "These summaries will be embedded and used to retrieve the raw image. ",
            "Give a concise summary of the image that is well optimized for retrieval.",
        ),
    }
    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }
    content_parts.append(text_part)
    content_parts.append(image_part)
    prompt = [HumanMessage(content=content_parts)]
    description = llm.invoke(prompt)
    return description.content
