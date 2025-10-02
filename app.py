import os
import re
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "korquad-rag-index")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

EMBEDDING_MODEL_ID = "BAAI/bge-m3"
LLM_MODEL_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
COHERE_RERANK_MODEL = "rerank-multilingual-v3.0"

TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5  # 5개로 복구
RELEVANCE_THRESHOLD = 0.35
MIN_SCORE_FOR_CONTEXT = 0.5

compression_retriever = None
model_obj = None
tokenizer_obj = None

def load_models():
    global compression_retriever, model_obj, tokenizer_obj

    if not all([PINECONE_API_KEY, HUGGING_FACE_TOKEN, COHERE_API_KEY]):
        raise RuntimeError("필수 환경 변수가 설정되지 않았습니다.")

    print("[초기화] 임베딩 모델 로드...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs={'device': 'cuda', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("[초기화] Pinecone 연결...")
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        raise RuntimeError(f"인덱스 '{PINECONE_INDEX_NAME}'가 존재하지 않습니다.")
    
    index = pc.Index(PINECONE_INDEX_NAME)
    vectorstore = PineconeVectorStore(  # LangchainPinecone → PineconeVectorStore
        index=index,
        embedding=embedding_model,
        text_key='text'
    )
    
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RETRIEVAL}
    )

    print("[초기화] Cohere Reranker 설정...")
    compressor = CohereRerank(
        model=COHERE_RERANK_MODEL,
        top_n=TOP_K_RERANK,
        cohere_api_key=COHERE_API_KEY
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    print("[초기화] LLM 로드...")
    model_obj = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_auth_token=HUGGING_FACE_TOKEN
    )
    tokenizer_obj = AutoTokenizer.from_pretrained(
        LLM_MODEL_ID,
        use_auth_token=HUGGING_FACE_TOKEN
    )
    
    print("[초기화] 완료!\n")

app = FastAPI(title="KorQuAD RAG API")

@app.on_event("startup")
def startup_event():
    load_models()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    retrieved_document_id: str
    retrieved_document: str
    question: str
    answers: str

def check_answer_validity(answer: str, context: str) -> bool:
    """사용 안함 - 후처리 검증 제거됨"""
    pass

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    global compression_retriever, model_obj, tokenizer_obj

    if not all([compression_retriever, model_obj, tokenizer_obj]):
        raise HTTPException(status_code=503, detail="서비스 초기화 중입니다.")
    if not request.question:
        raise HTTPException(status_code=400, detail="질문이 비어있습니다.")

    try:
        print(f"\n[질문] {request.question}")

        reranked_docs = compression_retriever.invoke(request.question)
        print(f"[검색] {len(reranked_docs)}개 문서 발견")
        
        if not reranked_docs:
            return {
                "retrieved_document_id": "N/A",
                "retrieved_document": "",
                "question": request.question,
                "answers": "관련 문서를 찾을 수 없습니다."
            }

        scores = [doc.metadata.get('relevance_score', 0) for doc in reranked_docs]
        best_score = scores[0]
        avg_score = sum(scores) / len(scores)
        score_std = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5
        
        print(f"[Rerank] 최고: {best_score:.4f}, 평균: {avg_score:.4f}, 표준편차: {score_std:.4f}")
        
        for idx, (doc, score) in enumerate(zip(reranked_docs, scores)):
            doc_id = doc.metadata.get('original_id', 'N/A')
            doc_title = doc.metadata.get('title', 'N/A')
            doc_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  [{idx+1}] ID={doc_id}, Score={score:.4f}, Title={doc_title}")
            print(f"      Preview: {doc_preview}...")

        best_doc = reranked_docs[0]
        relevance_score = best_score

        if relevance_score < RELEVANCE_THRESHOLD:
            print(f"[범위제한] 최고 점수 낮음 ({relevance_score:.4f} < {RELEVANCE_THRESHOLD})")
            return {
                "retrieved_document_id": "N/A",
                "retrieved_document": "",
                "question": request.question,
                "answers": "질문과 관련된 문서를 찾을 수 없습니다."
            }

        # 평균 점수 체크: 최고 점수가 높으면(0.5 이상) 평균 무시
        if relevance_score < 0.5 and avg_score < 0.25:
            print(f"[범위제한] 최고 점수도 낮고({relevance_score:.4f}), 평균도 낮음({avg_score:.4f})")
            return {
                "retrieved_document_id": "N/A",
                "retrieved_document": "",
                "question": request.question,
                "answers": "질문과 관련된 문서를 찾을 수 없습니다."
            }
        
        # 점수 격차 체크: 최고 점수가 높으면(0.5 이상) 격차 무시
        if len(scores) > 1 and relevance_score < 0.5:
            score_gap = best_score - scores[1]
            if score_gap > 0.7 and best_score < 0.5:
                print(f"[범위제한] 점수 격차 매우 큼 ({score_gap:.4f}), 신뢰도 낮음")
                return {
                    "retrieved_document_id": "N/A",
                    "retrieved_document": "",
                    "question": request.question,
                    "answers": "질문과 관련된 문서를 찾을 수 없습니다."
                }

        best_metadata = best_doc.metadata
        retrieved_document_id = best_metadata.get('original_id', best_metadata.get('id', 'N/A'))
        retrieved_document = best_metadata.get('full_context', best_doc.page_content)

        # 5개 문서 모두 사용 (점수별 가중치)
        contexts = []
        for idx, (doc, score) in enumerate(zip(reranked_docs, scores)):
            doc_content = doc.metadata.get('full_context', doc.page_content)
            
            # 점수에 따라 길이 조정
            if score > 0.8:
                max_len = 1500
            elif score > 0.6:
                max_len = 1000
            else:
                max_len = 500
            
            contexts.append(doc_content[:max_len])
        
        combined_context = "\n\n".join(contexts)[:3500]
        print(f"[컨텍스트] 총 {len(combined_context)}자 사용")

        prompt = f"""You are an AI that answers based solely on the given documents.

### Critical Rules:
1. Use ONLY the content from the [Document] below
2. If the information is not in the document, you MUST answer "The information is not available in the document"
3. Do NOT guess or use external knowledge
4. Answer concisely in 1-2 sentences

### Examples:
Document: Hangeul was created by King Sejong in 1446.
Question: Who created Hangeul?
Answer: King Sejong.

Document: Seoul is the capital of South Korea.
Question: What is the capital of Japan?
Answer: The information is not available in the document.

---

[Document]
{combined_context}

[Question]
{request.question}

[Answer]"""

        inputs = tokenizer_obj(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model_obj.device)

        with torch.no_grad():
            output_ids = model_obj.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                temperature=0.01,
                top_p=0.9,
                repetition_penalty=1.3,
                pad_token_id=tokenizer_obj.pad_token_id or tokenizer_obj.eos_token_id,
                eos_token_id=tokenizer_obj.eos_token_id
            )
        
        input_len = inputs["input_ids"].shape[1]
        answer_text = tokenizer_obj.decode(
            output_ids[0, input_len:],
            skip_special_tokens=True
        ).strip()

        if answer_text:
            answer_text = answer_text.split('\n')[0].split('.')[0].strip()
            
            for pattern in ['질문:', '답변:', '문서:', '[답변]', '[질문]', '###']:
                if pattern in answer_text:
                    answer_text = answer_text.split(pattern)[-1].strip()
            
            if answer_text and not answer_text.endswith('.'):
                answer_text += '.'
            
            if len(answer_text) > 200:
                answer_text = answer_text[:197] + '...'
            
            final_answer = answer_text if answer_text else "문서에 해당 정보가 없습니다."
        else:
            final_answer = "문서에 해당 정보가 없습니다."

        # 간단한 검증만: 부정 답변 체크
        negative_keywords = ['정보가 없습니다', '찾을 수 없습니다', '알 수 없습니다', 'not available', 'cannot find']
        if any(kw in final_answer for kw in negative_keywords):
            final_answer = "문서에 해당 정보가 없습니다."

        print(f"[답변] {final_answer}\n")

        return {
            "retrieved_document_id": str(retrieved_document_id),
            "retrieved_document": retrieved_document,
            "question": request.question,
            "answers": final_answer
        }

    except Exception as e:
        print(f"[오류] {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류: {str(e)}")

@app.get("/")
def read_root():
    return {
        "service": "KorQuAD RAG API",
        "embedding_model": EMBEDDING_MODEL_ID,
        "llm_model": LLM_MODEL_ID,
        "rerank_model": COHERE_RERANK_MODEL
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if all([compression_retriever, model_obj, tokenizer_obj]) else "initializing"
    }