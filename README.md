# Illuminus's Assignment
This is my entry for Illuminus's position as an AI Engineer in NLP.

## 1. Setup

### 1.1 Dependencies

It's recommended to use a Python 3.11 virtual environment. Below are two ways to setup dependencies:

#### For **conda** user:

The command below will create a conda environment named `illuminus`, you can change this in `environment.yml`

```sh
conda env create -f environment.yml
conda activate illuminus
```

#### For **pip** user:

Make sure you are running on python 3.11.

```sh
pip install -r requirements.txt
```

### 1.2 Ollama
This project uses Ollama as the LLM backend, which depends on the host computer hardware.

Windows and Macos users: Download and Install at https://ollama.com/download

For Linux:
```
curl -fsSL https://ollama.com/install.sh | sh
```

Note: Ollama will use port http://localhost:11434 by default.

#### Download LLM models:

I use `llava-llama3` as the main llm for this project. This model can handle iamges but requires atleast 16GB of RAM. If you have a GPU with less than 16GB, Ollama will offload the model to CPU.

```
ollama pull llava-llama3
```

I have also tested on `phi3` and `llava-phi3`, the model ran fast but the output did not match my expectation.

### 1.3 Prepare vectorstore for RAG:

The provided documents are stored in `docs` folder. This script will convert those documents into vertorstore.

```sh
python scripts/prepare_dataset.py
```

## 2. Running the App

Run the app:
```sh
chainlit run app.py
```