from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from FlagEmbedding import FlagAutoReranker, FlagModel
from langchain.prompts.chat import ChatPromptTemplate
####
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import os,sys
import pandas as pd
import numpy as np
import random


data_file = "merged_data.csv"
local_path = "../bge-reranker-large"
faiss_index_path = "faiss_index"

def setup_models():
    generation_model = ChatOllama(model="qwen2.5:72b",
                                    temperature=0,
                                    seed=42)
    embedding_model = OllamaEmbeddings(model="bge-m3")
    reranker = FlagAutoReranker.from_finetuned(local_path, query_max_length=256, passage_max_length=512, use_fp16=True, devices=['cuda:0'])
    return generation_model, embedding_model, reranker


def setup_retriever(documents, embedding_model, faiss_index_path):
    if os.path.exists(faiss_index_path):
        vectorstore = FAISS.load_local(faiss_index_path, embedding_model,allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_texts(documents, embeddings=embedding_model)
        vectorstore.save_local(faiss_index_path)
    return vectorstore.as_retriever()

def setup_generation_chain(retriever, generation_model):
    prompt_template = ChatPromptTemplate.from_template(
        "You are an assistant. Answer the question based on the following contexts. If you don't know the answer, just only say \"No Answer Present\". Use two sentences maximum and keep the answer concise.\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:"
    )
    
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | generation_model
        | StrOutputParser()
    )



def rerank_documents(query,retrieved_docs,top_K):

    query_doc_pairs = [[query, doc.page_content] for doc in retrieved_docs]
    scores = reranker.compute_score(query_doc_pairs, normalize=True)
    scored_docs = list(zip(retrieved_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in scored_docs[:top_K]]
   
    return reranked_docs

def advanced_rag(query, top_K, random_rank):
    #retrieved_docs = retriever.get_relevant_documents(query,k=top_K)
    #response_navie = generation_chain.invoke({"question": query, "context": retrieved_docs})
    pre_retrieved_docs = retriever.get_relevant_documents(query,k=100)
    reranked_docs = rerank_documents(query, pre_retrieved_docs, top_K)
    doc_context_list = []
    if random_rank is False:
        for idx, doc in enumerate(reranked_docs, start=1):
            x = f"{idx}. {doc.page_content}"
            doc_context_list.append(x)

        context_str = "\n".join(doc_context_list)
    else:
        random.shuffle(reranked_docs)
        for idx, doc in enumerate(reranked_docs, start=1):
            x = f"{idx}. {doc.page_content}"
            doc_context_list.append(x)
        context_str = "\n".join(doc_context_list)
    response = generation_chain.invoke({"question": query, "context": context_str})

    return response,context_str

def load_documents(file_path):
    df = pd.read_csv(file_path)
    documents = df['passage_text'].tolist()
    querys = df['query'].tolist()
    answers = df['answer'].tolist()
    return documents, querys, answers

documents, querys, answers = load_documents(data_file)
generation_model, embedding_model, reranker = setup_models()
retriever = setup_retriever(documents, embedding_model,faiss_index_path)
generation_chain = setup_generation_chain(retriever, generation_model)
top_K = 5

responses = []
uniq_data = dict(zip(querys[:500],answers[:500]))
for query, answer in uniq_data.items():
    response,context_str = advanced_rag(query,top_K, random_rank=False)
    response_rand,_ = advanced_rag(query, top_K,random_rank=True)
    response_rand2,_ = advanced_rag(query,top_K,random_rank=True)
    response_rand3,_ = advanced_rag(query,top_K,random_rank=True)
    responses.append({
        "query":query,
        "response":response,
        "response_rand":response_rand,
        "response_rand2":response_rand2,
        "response_rand3":response_rand3,
        "context":context_str,
        "answer":answer})

df_responses = pd.DataFrame(responses)
csv_file_path = "response_test.csv"
df_responses.to_csv(csv_file_path, index=False)
sys.exit(0)



rouge = Rouge()
#candidate_navie = response_navie.lower()
candidate_rerank = response.lower()
reference = "Cinnamon, Sprinkle, Prickly Pear\/ Nopal, Grapefruit, carbohydrates, certain vitamins.".lower()


#scores_navie = rouge.get_scores(candidate_navie, reference)
scores_rerank = rouge.get_scores(candidate_rerank, reference)

#print("NAVIE ROUGE-1:", scores_navie[0]["rouge-1"])
#print("NAVIE ROUGE-2:", scores_navie[0]["rouge-2"])
#print("NAVIE ROUGE-L:", scores_navie[0]["rouge-l"])


print("RERANK ROUGE-1:", scores_rerank[0]["rouge-1"])
print("RERANK ROUGE-2:", scores_rerank[0]["rouge-2"])
print("RERANK ROUGE-L:", scores_rerank[0]["rouge-l"])

#bleu_candidate_navie = response_navie.lower()
bleu_candidate_rerank = response.lower()
reference = "Cinnamon, Sprinkle, Prickly Pear\/ Nopal, Grapefruit, carbohydrates, certain vitamins.".lower()

# 将文本分词
#candidate_tokens_navie = bleu_candidate_navie.split()
candidate_tokens_rerank = bleu_candidate_rerank.split()
reference_tokens = [ref.split() for ref in reference]

# 计算 Bleu-1
#navie_bleu_1 = sentence_bleu(reference_tokens, candidate_tokens_navie, weights=(1, 0, 0, 0))
rerank_bleu_1 =sentence_bleu(reference_tokens, candidate_tokens_rerank, weights=(1, 0, 0, 0))
#print(f"Navie Bleu-1 Score: {navie_bleu_1}")
print(f"Rerank Bleu-1 Score: {rerank_bleu_1}")
