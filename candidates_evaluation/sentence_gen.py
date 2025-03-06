from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import pandas as pd

def triple2sentence(head:str, relation:str, tail:str) -> str:
    """
    Transform a triple into a sentence.
    """
    template = """Transform the following triples into a sentence: {head} {relation} {tail}

    Answer: Give the sentence."""

    prompt = ChatPromptTemplate.from_template(template)
    llm = Ollama(model="mistral")
    model = llm
    chain = prompt | model

    sentence = chain.invoke({"head": head,
                "relation": relation,
                'tail': tail}
                )
   
    return sentence

def df2sentence_list(df:pd.DataFrame)-> list[str]:
    """
    Turn a dataframe into a list of sentence.
    The list of sentence is sorted in the same order as the sentence.
    """
    sentence_list = []
    for item in df.iterrows():
        triple = item[1]
        # extract head, relation, tail
        head = triple['Head']
        relation = triple['Relation']
        tail = triple['Tail']
        # transform triple into sentences
        sentence = triple2sentence(head,relation,tail)
        sentence_list.append(sentence)
    return sentence