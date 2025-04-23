from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd

def create_retriever(file_path:str, top_k:int = 2):
    loader = CSVLoader(file_path=file_path)
    docs = loader.load()
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)
    # Instantiate the embedding model
    embedder = HuggingFaceEmbeddings()
    # Create the vector store 
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    return retriever

def prompt_answer(prompt_template:str, **kwargs) -> str:
    # Define llm
    llm = Ollama(model="mistral", temperature=0.1)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = llm
    chain = prompt | model
    result = chain.invoke(kwargs)
    return result

def plain_triple(df:pd.DataFrame):
    template = """
    1. Evaluate if the triple represent a correct fact or not.\n
    2. Is the triple correct: answer "1" if it is correct and "0" otherwise.\n
    3. If you don't know the answer, just say that "-1"\n
    4. Start the answer with 'Score:'\n
    5. A triple represent a relation between the head entity and the tail entity\n
    Triple: {triple}
    Helpful Answer:"""

    triple_list = []
    for item in df.iterrows():
        triple = item[1]
        head = triple['Head']
        relation = triple['Relation']
        tail = triple['Tail']
        triple_sentence = f"Head:{head}\t Relation:{relation}\t Tail:{tail}"
        triple_list.append(triple_sentence)

    score_list = []
    for triple in triple_list:
        score = prompt_answer(template, triple=triple)
        score_list.append(score)
    return score_list

def RAG_triple(df:pd.DataFrame, retriever):
    template = """
    1. Use the following pieces of context to determine if the final triple present correct fact or not.\n
    2. Is the triple correct: answer "1" if it is correct and "0" otherwise.\n
    3. If you don't know the answer, just say that "-1"\n
    4. Start the answer with 'Score:'\n
    5. A triple represent a relation between the head entity and the tail entity\n

    Here similar triples to help you make a decision:
    Context: {context}
    Here the triple to evaluate:
    Triple: {triple}

    Helpful Answer:"""

    def get_context_list(triple):
        new_context_list = []
        context_list =  retriever.invoke(triple)
        for context in context_list:
            context = context.page_content
            new_context_list.append(context)
        return new_context_list
    
    triple_list = []
    for item in df.iterrows():
        triple = item[1]
        head = triple['Head']
        relation = triple['Relation']
        tail = triple['Tail']
        triple_sentence = f"Head:{head}\t Relation:{relation}\t Tail:{tail}"
        triple_list.append(triple_sentence)

    score_list = []
    for triple in triple_list:
        context = get_context_list(triple)
        score = prompt_answer(template, triple=triple, context=context)
        score_list.append(score)

    return score_list

def plain_sentence(df:pd.DataFrame):
    return None



