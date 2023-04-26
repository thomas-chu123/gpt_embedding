import openai
import os
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import PyPDFLoader  # for loading the pdf
from langchain.chains import ChatVectorDBChain # for chatting with the pdf

openai.organization = "org-BFzxcJ0l6gshdj1KE9M2fPvN"
openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"


def create_embedding():
    # loader = DirectoryLoader("./source/data.txt", glob="data.txt")
    # txt_docs = loader.load_and_split()
    # txt_doc = UnstructuredFileLoader("./source/data.txt").load()
    with open("./source/data.txt") as f:
        txt_doc = f.read()

    loader = PyPDFLoader("./source/Wi-Fi_EasyMesh_Specification_v3.pdf")
    pages = loader.load_and_split()

    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # texts = text_splitter.split_text(txt_doc)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(pages, embeddings, persist_directory="db")
    docsearch.persist()
    docsearch = None
    print("created embedding db is done")

def search_embedding(user_input):
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma(persist_directory="db", embedding_function=embeddings)
    # chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff",
    #                                                    retriever=docsearch.as_retriever())

    chain = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
                                        docsearch, return_source_documents=True)

    # user_input = input("What's your question: ")
    result = chain({"question": user_input, "chat_history": ""})
    print("Answer: " + result["answer"].replace('\n', ' '))
    print("Source: " + result["sources"])

    # Going with option 1 (txt files)
    # Create embeddings
    # embeddings = OpenAIEmbeddings(openai_key=openai.api_key)
    # Write in DB
    # txt_docsearch = Chroma.from_documents(txt_doc, embeddings)

    # Define LLM
    # llm = ChatOpenAI(model_name="embedded-a", temperature=0.2)
    # Create Retriever
    # In case answers are cut-off or you get error messages (token limit)
    # use different chain_type
    # qa_txt = RetrievalQA.from_chain_type(llm=llm,
    #                                    chain_type="stuff",
    #                                    retriever=txt_docsearch.as_retriever())

    # query = "login web gui and create firewall rule"
    # qa_txt.run(query)

def main():
    # create_embedding()
    search_embedding("list down the field in Steering Policy TLV format ")

if __name__ == "__main__":
    main()