# Assignment Report

Author: Tien Duong \
Date: Thu Oct  3 22:01:24 +07 2024


## 1. Overview

This project is a conversational Retrieval-Augmented Generation (RAG) chatbot
that plays the role of "Choi," a character who interacts with the user as if the
user is his father, David, who has Alzheimer's disease. The chatbot's purpose is
to provide a personalized and empathetic conversational experience based on
Choi's background. It combines an LLM-based dialogue system with retrieved
context to generate relevant and informative responses.


## 2. Architecture

The chatbot is built with the following components:
- Langchain v0.3, as the LLM framework.
- chainlit, as the UI.
- Ollama, as the LLM backend
- Chroma, as the vectorstore.

I chose `Ollama` as the backend because I cannot afford OpenAI API and it is the most familiar framework to me. On the flip side, this open more opportunities for opensource LLMs, with finetuned models for Korean language. The entire project is built around Ollama API.

The project is structured as follow:

```
illuminus-assignment
├── app.py
├── db/
│   └── ...
├── docs/
│   ├── Game Rules for _Which One is It__.txt
│   ├── Who are You_.txt
│   └── 너는 누구니.txt
├── qachatbot
│   ├── bot
│   │   ├── __init__.py
│   │   ├── bot.py
│   │   ├── chat.py
│   │   └── vision.py
│   ├── settings.py
│   └── utils
│       ├── __init__.py
│       └── vectorstore.py
├── report
│   ├── REPORT.md
│   └── rag_retrieval_generation.png
├── samples
│   ├── sample.txt
│   └── sample_pic.jpg
└── scripts
    └── prepare_dataset.py
```

- `app.py`: handle the UI and frontends.
- `qachatbot`: the main logic behind the chatbot.
    - `bot`: contain chatbot logic, how it's set-up.
    - `settings.py`: stores global vectorstore, constants and variable.
    - `utils`: helper class vectorstore.
- `report`: The report for this assignment.
- `samples`: samples input for the chatbot.
- `scripts`: the scripts to create the vectorstore, it's required to run this before running `app.py`. 

### LLM selection

As hinted in the _Which One is it_ document, the LLM should be able to handle images. Therefore, I chose `llava-llama3`, a multimodal-LLM, to handle the entire chatbot system.

I also tried `phi3`-based model, it is light-weight and fast, but cannot handle large context.

## 3. Methodology

### RAG

A simple RAG pipeline is implemented. RAG is used to retrieve the provided data to enhance the chatbot generation output. In this case, it act as Choi's memory.

#### Data preparation

The provided documents are stored in `docs/` directory.

```
docs
├── .unused
├── Game Rules for _Which One is It__.txt
├── Who are You_.txt
└── 너는 누구니.txt
```

To enhance the model, I also create synthetic data by generating short converations between David and Choi, but the model output results did not do well as expected, I scraped the idea and archive them in the `.unused` directory.

These documents are then read, splitted by chunks of 1000 words and embedded using `BAAI/bge-small-en-v1.5` model. Which is then persist on disk using Chorma.

The vectorstore is stored globally since history is stored in-memory per session.

```sh
python scripts/prepare_dataset.py
```

### Chatbot

The diagram below show how the chatbot works with text input:

![RAG Flow](rag_retrieval_generation.png)

For image, because Langchain does not handle History and Image well, here is the workaround I implemented:
1. The image is first interpreted by the LLM to extract the description.
2. The description is then used as the context to generate new response for the user.
3. Only the description is saved on the memory.

Because of this, LLM is run twice before generating output to the user.

## 4. Retro

Here are the challenges that I faced:
- I find it very challenge to work on this assignment with a full-time job, since I have so little time left after work to work on it. Some corners were cut:
  - No deployment/serving
  - No testings/evaluation
  - The chatbot is at bare minimum.
- Langchain is fast, but over abstractions make customization really hard.
- The current system will always retrieve data from the vectorstore.
  - This can be solved using chain-of-thought and tools method to determine if retrieval is needed or not. 
