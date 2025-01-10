import argparse
import os

from langchain_text_splitters import Language
from langchain_text_splitters.markdown import RecursiveCharacterTextSplitter
import ollama
import pymupdf
import pymupdf4llm
from pymilvus import MilvusClient
from tqdm import tqdm


ANSWER_MODEL = "llama3.1"
EMBEDDING_MODEL = "hf.co/second-state/jina-embeddings-v2-base-de-GGUF:Q5_K_M"
EMBEDDING_DIM = 768


def embed_text(text):
    """ Compute a vector embedding of the given text.

    :param text: list of text chunks to embed
    :return: vector embeddings
    """
    return ollama.embed(EMBEDDING_MODEL, text).embeddings


def generate_chunks(md_text):
    """ Split markdown text into chunks of bounded length by recursively splitting at certain symbols.

    :param md_text:
    :return: list of text chunks
    """
    # Define custom separator symbols based on pdf table of contents
    custom_separators = []
    headers = ["[I-V]{1,3}\.", "[0-9]+\.", "[a-z]\)", "[a-z]+\)", "[0-9]+"]
    for header in headers:
        custom_separators.append(f"\n\n\*\*{header}\*\* ")
        custom_separators.append(f"\n\n{header} ")
    # Use custom separator symbols along with standard markdown separators to split the text
    separators = custom_separators + RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                              chunk_overlap=128,
                                              separators=separators,
                                              is_separator_regex=True)
    return splitter.split_text(md_text)


def init_db(milvus_client, collection_name, pdf_dir):
    """ Create a collection in the database that contains the text extracted from the pdf files in the given directory.

    :param milvus_client: database client
    :param collection_name: name of collection to be created
    :param pdf_dir: path to directory with pdf files
    :return: None
    """
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=EMBEDDING_DIM,
        metric_type="IP"
    )
    for doc in tqdm(os.listdir(pdf_dir)):
        doc_path = os.path.join(pdf_dir, doc)
        try:
            md_text = pymupdf4llm.to_markdown(doc_path, show_progress=False)
        except:
            print(f"Skipping path <{doc_path}>.")
            continue

        # We use the modification date of the pdf to date the information contained in the pdf
        # This is a hack and not reliable - should instead extract date directly from the pdf content
        mod_date = pymupdf.open(doc_path).metadata["modDate"]
        mod_date_ddmmyyyy = ".".join([mod_date[8:10], mod_date[6:8], mod_date[2:6]]) # DD.MM.YYYY format
        chunks = generate_chunks(md_text)
        data = [{"id": i, "vector": v, "text": t, "source": doc, "date": mod_date_ddmmyyyy}
                for i, (t, v) in enumerate(zip(chunks, embed_text(chunks)))]
        milvus_client.insert(collection_name=collection_name, data=data)


def init_rag_pipeline(pdf_dir):
    """ Initialize everything that is needed for the RAG tasks. Download the embedding and answer models if necessary.
    Create the vector db or load it if it already exists.

    :param pdf_dir: path to directory with pdf files that serve as knowledge base for RAG
    :return: database client and name of the collection in the database
    """
    print("Downloading LLMs for embedding and generation. This might take a while.")
    ollama.pull(EMBEDDING_MODEL)
    ollama.pull(ANSWER_MODEL)

    print("Setting up vector db.")
    db_path = os.path.join(pdf_dir, "milvus.db")
    milvus_client = MilvusClient(uri=db_path)
    collection_name = "documents"
    if milvus_client.has_collection(collection_name):
        print(f"Found existing db at <{db_path}>.")
    else:
        print(f"Initializing new vector db at <{db_path}>. It might take a while to compute the embeddings.")
        init_db(milvus_client, collection_name, pdf_dir)
    return milvus_client, collection_name


def generate_prompt(question, context, context_date):
    """ Generate the full prompt for the LLM.

    :param question: question posed by user
    :param context: related context retrieved from the vector db
    :param context_date: date of retrieved context pieces
    :return: full prompt with instructions, question and context
    """
    context = "\n\n".join([f"Kontext vom {cd}: {c}" for c, cd in zip (context, context_date)])
    prompt = ("Deine Aufgabe ist es, eine Frage auf Basis des gegebenen Kontexts zu beantworten.\n"
              "Du verwendest ausschließlich die im Kontext enthaltenen Informationen, um die Frage zu beantworten.\n"
              "Du gibst eine direkte Antwort auf die Frage.\n\n"
              f"Frage: {question}\n\n"
              f"{context}")
    return prompt


def main(pdf_dir, debug):
    """ Main RAG loop that takes a question from the user and generates an answer.

    :param pdf_dir: path to directory with pdf files that serve as knowledge base for RAG
    :param debug: If true, will display the full prompt together with the answer
    :return: None
    """
    milvus_client, collection_name = init_rag_pipeline(pdf_dir)
    while True:
        # Receive question from the user
        question = input("\nBitte stelle eine Frage. (\exit to quit)\n")
        if question == "\exit":
            break
        elif question == "":
            continue

        # Retrieve relevant context from vector db
        search_results = milvus_client.search(
            collection_name=collection_name,
            data=embed_text([question]),
            output_fields=["text", "date"],
            limit=11)
        context = [result["entity"]["text"] for result in search_results[0]]
        context_date = [result["entity"]["date"] for result in search_results[0]]

        # Generate an answer to the question
        prompt = generate_prompt(question, context, context_date)
        response = ollama.chat(model="llama3.1",
                               messages=[{'role': 'user', 'content': prompt}])
        if debug:
            print(prompt, "\n\n")
        print(response["message"]["content"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", type=str, help="Path to directory with pdf files")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="Display the full prompt")
    args = parser.parse_args()
    main(args.pdf_dir, args.debug)