from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from FlagEmbedding import FlagModel,FlagReranker
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
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
import os,sys,csv
import pandas as pd
import numpy as np
from TripletRewrite import KGRewriter

os.environ["SERPER_API_KEY"] = "5b0b25fce3ab6db7d03cb97dadaabc232f11fdae"
search = GoogleSerperAPIWrapper(k=5)


data_file = "../merged_data.csv"
local_path = "../bge-reranker-large"
faiss_index_path = "../faiss_index"
graph_index_path = "../graph_index"
graph_file = "../graphTriplet.csv"
top_K = 10
eval_query =2

def setup_models():
    generation_model = ChatOllama(model="qwen2.5:72b",temperature=0)
    embedding_model = OllamaEmbeddings(model="bge-m3")
    #reranker = FlagAutoReranker.from_finetuned(local_path, use_fp16=True, devices=['cuda:1'])
    reranker = FlagReranker(local_path, use_fp16=True)
    return generation_model, embedding_model, reranker


def setup_retriever(documents, embedding_model, faiss_index_path, graph_index_path,graph_docs):
    if os.path.exists(faiss_index_path):
        vectorstore = FAISS.load_local(faiss_index_path, embedding_model,allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_texts(documents, embeddings=embedding_model)
        vectorstore.save_local(faiss_index_path)
    
    #graph_retriever = BM25Retriever.from_texts(graph_docs,k=10)
    #return vectorstore.as_retriever(), graph_retriever
    #FAISS for Graph retriever
    if os.path.exists(graph_index_path):
        vectorstore_graph = FAISS.load_local(graph_index_path, embedding_model,allow_dangerous_deserialization=True)
    else:
        vectorstore_graph = FAISS.from_texts(graph_docs,embedding=embedding_model)
        vectorstore_graph.save_local(graph_index_path)
    return vectorstore.as_retriever(),vectorstore_graph.as_retriever()

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


def rerank_documents(query,merge_docs,top_K):

    #query_doc_pairs = [[query, doc.page_content] for doc in retrieved_docs]
    #scores = reranker.compute_score(query_doc_pairs, normalize=True)
    #scored_docs = list(zip(retrieved_docs, scores))
    #scored_docs.sort(key=lambda x: x[1], reverse=True)
    #reranked_docs = [doc for doc, score in scored_docs[:top_K]]
    
    #return reranked_docs
    doc_context_list=[]
    query_doc_pairs = [[query, doc] for doc in merge_docs]
    scores = reranker.compute_score(query_doc_pairs, normalize=True)
    scored_docs = list(zip(merge_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in scored_docs[:top_K]]
    for idx, doc in enumerate(reranked_docs, start=1):
        x = f"{idx}. {doc}"
        doc_context_list.append(x)
    context_str = "\n".join(doc_context_list)
    return reranked_docs,context_str


def advanced_rag(query, top_K, with_graph=True):
    pre_retrieved_docs = retriever.get_relevant_documents(query,k=50)
    if with_graph==True:
        retrieved_graph = graph_retriever.get_relevant_documents(query,k=10)
        #retrieved_graph = graph_retriever.invoke(query)
    else:
        retrieved_graph = []
    results_websearch = search.results(query)
    merge_docs = merge_context(pre_retrieved_docs,results_websearch,retrieved_graph)
    
    reranked_docs,context_str = rerank_documents(query, merge_docs, top_K)
    print("docs after rank:",context_str)
    response = generation_chain.invoke({"context": context_str, "question": query})

    return response


def merge_context(pre_retrieved_docs,results_websearch,retrieved_graph):
    merge_docs = []
    entities_set = set()
    description_set = set()
    # for val in retrieved_graph:
    #     relation = val.page_content
    #     for item in triplet_dict[relation]:
    #         formatted_string = f"{item[0]},{relation},{item[1]}"
    #         merge_docs.append(formatted_string)
    for val in retrieved_graph:
        relation = val.page_content
        for item in triplet_dict[relation]:
            entities_set.add(item[0])
            entities_set.add(item[1])
            description_set.add(relation)
            formatted_string = f"{item[0]},{relation},{item[1]}"
            print(formatted_string)

    kg_doc = kg_rewriter.Rewrite(list(entities_set),list(description_set))
    merge_docs.append(kg_doc)

    for doc in pre_retrieved_docs:
        merge_docs.append(doc.page_content)
    if "organic" in results_websearch:
        for i, result in enumerate(results_websearch["organic"]):
            title = result.get("title", "No title available")
            snippet = result.get("snippet", "No snippet available")
            merge_docs.append(f"Title:{title} Context:{snippet}")

    return merge_docs



def load_documents(file_path):
    df = pd.read_csv(file_path)
    documents = df['passage_text'].tolist()
    querys = df['query'].tolist()
    answers = df['answer'].tolist()
    return documents, querys, answers

def load_Graph(file_path):
    triplet_dict = {}  # 初始化词典

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            subject, predicate, object = row  # 解包每一行的三元组
            if predicate not in triplet_dict:
                triplet_dict[predicate] = []  # 如果 predicate 不存在，初始化一个空列表
            triplet_dict[predicate].append((subject, object))  # 将 (subject, object) 添加到对应的 predicate 列表中

    return triplet_dict 

documents, querys, answers = load_documents(data_file)
generation_model, embedding_model, reranker = setup_models()
triplet_dict = load_Graph(graph_file)
graph_docs = list(triplet_dict.keys())
retriever, graph_retriever = setup_retriever(documents, embedding_model, faiss_index_path, graph_index_path, graph_docs)
generation_chain = setup_generation_chain(retriever, generation_model)
kg_rewriter = KGRewriter(generation_model,triplet_dict)

responses = []
uniq_data = dict(zip(querys[:eval_query*10],answers[:eval_query*10]))
for query, answer in uniq_data.items():
    #response = advanced_rag(query,top_K,False)
    response="null"
    response_graph = advanced_rag(query,top_K)
    responses.append({
        "query":query,
        "response":response,
        "response_graph":response_graph,
        "answer":answer})

df_responses = pd.DataFrame(responses)
csv_file_path = "response_hyper_test.csv"
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
