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

def get_triple_sentence(triple):
    head = triple['Head']
    relation = triple['Relation']
    tail = triple['Tail']
    triple_sentence = f"Head:{head}\t Relation:{relation}\t Tail:{tail}"
    return triple_sentence

def get_triple_list(df):
    triple_list = []
    for item in df.iterrows():
        triple = item[1]
        triple_sentence = get_triple_sentence(triple)
        triple_list.append(triple_sentence)
    return triple_list

def plain_triple(df:pd.DataFrame):
    template = """
    1. Evaluate if the triple represent a correct fact or not.\n
    2. Is the triple correct: answer "1" if it is correct and "0" otherwise.\n
    3. If you don't know the answer, just say that "-1"\n
    4. Start the answer with 'Score:'\n
    5. A triple represent a relation between the head entity and the tail entity\n
    Triple: {triple}
    Helpful Answer:"""

    triple_list = get_triple_list(df)

    score_list = []
    for index,triple in enumerate(triple_list):
        score = prompt_answer(template, triple=triple)
        score_list.append(score)
        if index%100 == 0:
            print(f'{index} / {len(triple_list)}')
    return score_list

def context_triple(evaluation_df:pd.DataFrame, original_df:pd.DataFrame):
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
    context_list =  []
    triple_list = []
    for item in evaluation_df.iterrows():
        context_triple = []
        triple = item[1]
        head = triple['Head']
        relation = triple['Relation']
        tail = triple['Tail']
        triple_sentence = f"Head:{head}\t Relation:{relation}\t Tail:{tail}"
        context_triple.append(original_df[original_df['Head'] == head].sample(1))
        context_triple.append(original_df[original_df['Relation'] == relation].sample(1))
        context_triple.append(original_df[original_df['Tail'] == tail].sample(1))
        context_list.append(context_triple)
        triple_list.append(triple_sentence)

    score_list = []
    for index, triple in enumerate(triple_list):
        context = context_list[index]
        score = prompt_answer(template, triple=triple, context=context)
        score_list.append(score)
        if index%100 == 0:
            print(f'{index} / {len(triple_list)}')
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
    
    triple_list = get_triple_list(df)
    score_list = []
    for index, triple in enumerate(triple_list):
        context = get_context_list(triple)
        score = prompt_answer(template, triple=triple, context=context)
        score_list.append(score)
        if index%100 == 0:
            print(f'{index} / {len(triple_list)}')

    return score_list

def triple2sentence(triple):
    template = """
    1. Your job is only to translate triple into sentence, no matter if it is correct or not
    2. A triple is two entities (head and tail) linked by a relation
    3. Transform the following triples into a sentence: '{triple}'
    4. If the triple present incorrect fact, still translate this as it is
    5. Do not make negative sentence 
    Answer: Give the sentence."""

    sentence = prompt_answer(template, triple=triple)
    return sentence

def get_sentence_list(df):
    sentence_list = []
    for item in df.iterrows():
        triple = item[1]
        triple_sentence = get_triple_sentence(triple)
        sentence = triple2sentence(triple_sentence)
        sentence_list.append(sentence)
    return sentence_list

def plain_sentence(df:pd.DataFrame):
    template = """
    1. Evaluate if the sentence represent a correct fact or not.\n
    2. Is the triple correct: answer "1" if it is correct and "0" otherwise.\n
    3. If you don't know the answer, just say that "-1"\n
    4. Start the answer with 'Score:'\n

    Sentence: {sentence}

    Helpful Answer:"""
    sentence_list = get_sentence_list(df)
    score_list = []
    for index, sentence in enumerate(sentence_list):
        score = prompt_answer(template, sentence=sentence)
        score_list.append(score)
        if index%100 == 0:
            print(f'{index} / {len(sentence_list)}')
    return score_list

def RAG_sentence(df, retriever):
    template = """
    1. Use the following pieces of context to determine if the sentence represent correct fact or not.\n
    2. Is the sentence stating correct facts: answer "1" if it is correct and "0" otherwise.\n
    3. If you don't know the answer, just say that "-1"\n
    4. Start the answer with 'Score:'\n

    Context: {context}

    Sentence: {sentence}

    Helpful Answer:"""

    def get_context_list(sentence):
        new_context_list = []
        context_list =  retriever.invoke(sentence)
        for context in context_list:
            context = context.page_content
            context = triple2sentence(context)
            new_context_list.append(context)
        return new_context_list

    sentence_list = get_sentence_list(df)
    score_list = []
    for index, sentence in enumerate(sentence_list):
        context = get_context_list(sentence)
        score = prompt_answer(template, sentence=sentence, context=context)
        score_list.append(score)
        if index%100 == 0:
            print(f'{index} / {len(sentence_list)}')
    return score_list

def context_sentence(evaluation_df:pd.DataFrame, original_df:pd.DataFrame):
    template = """
    1. Use the following pieces of context to determine if the sentence represent correct fact or not.\n
    2. Is the sentence stating correct facts: answer "1" if it is correct and "0" otherwise.\n
    3. If you don't know the answer, just say that "-1"\n
    4. Start the answer with 'Score:'\n

    Context: {context}

    Sentence: {sentence}

    Helpful Answer:"""
    context_list = []
    for item in evaluation_df.iterrows():
        context_triple = []
        triple = item[1]
        head = triple['Head']
        relation = triple['Relation']
        tail = triple['Tail']
        head_cont = original_df[original_df['Head'] == head].sample(1)
        for id, elem in head_cont.iterrows():
            elem_sent = triple2sentence(elem)
            context_list.append(elem_sent)
        rel_cont = original_df[original_df['Relation'] == relation].sample(1)
        for id, elem in rel_cont.iterrows():
            elem_sent = triple2sentence(elem)
            context_list.append(elem_sent)
        tail_cont = original_df[original_df['Tail'] == tail].sample(1)
        for id, elem in tail_cont.iterrows():
            elem_sent = triple2sentence(elem)
            context_list.append(elem_sent)

    sentence_list = get_sentence_list(evaluation_df)
    score_list = []
    for index, sentence in enumerate(sentence_list):
        context = context_list[index]
        score = prompt_answer(template, sentence=sentence, context=context)
        score_list.append(score)
        if index%100 == 0:
            print(f'{index} / {len(sentence_list)}')
    return score_list




