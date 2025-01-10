# Prerequisites
* Python 3.9 (Other python versions like 3.12 might cause a bug with Milvus)
* Install ollama [from here](https://ollama.com/download) or `sudo snap install ollama`
* Install the required libraries `pip install -r requirements.txt`

# Running the code
* Put pdf files in a directory, e.g., `./docs`. Then run `python ./src/main.py ./docs` for RAG.
First start will be a bit slow because LLMs need to be downloaded and the db needs to be initialized.