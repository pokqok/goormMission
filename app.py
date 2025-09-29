import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pinecone import Pinecone as PineconeClient

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# FastAPI 애플리케이션 초기화
app = FastAPI()

# --- 모델 및 벡터 스토어 로드 (서버 시작 시 1회 실행) ---
embedding_model = None
vector_store = None
llm_pipeline = None
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "korquad-rag-index"

@app.on_event("startup")
async def startup_event():
    """
    서버 시작 시 모델과 Pinecone 벡터스토어를 로드합니다.
    """
    global embedding_model, vector_store, llm_pipeline

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")

    print("서버 시작: 임베딩 모델 및 Pinecone 벡터 스토어 로드를 시작합니다.")
    try:
        # 1. 임베딩 모델 로드 (KoBERT로 변경)
        embedding_model = HuggingFaceEmbeddings(
            model_name="skt/kobert-base-v1",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )

        # 2. 기존 Pinecone 인덱스 로드
        pc = PineconeClient(api_key=PINECONE_API_KEY)
        if INDEX_NAME not in pc.list_indexes().names():
            raise FileNotFoundError(f"Pinecone에서 '{INDEX_NAME}' 인덱스를 찾을 수 없습니다. data_processing.py를 먼저 실행해주세요.")
        
        vector_store = Pinecone.from_existing_index(INDEX_NAME, embedding_model)
        print("Pinecone 벡터 스토어 로드 완료.")

        # 3. LLM 모델 및 토크나이저 로드
        print("LLM 모델을 로드합니다 (EleutherAI/polyglot-ko-1.3b).")
        model_name = "EleutherAI/polyglot-ko-1.3b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        device = 0 if torch.cuda.is_available() else -1
        
        # 4. Transformers 파이프라인 생성
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=200,
            model_kwargs={"temperature": 0.7, "repetition_penalty": 1.2}
        )
        print("LLM 파이프라인 생성 완료. 서버가 준비되었습니다.")

    except Exception as e:
        print(f"초기화 중 심각한 오류 발생: {e}")


# --- API 요청 및 응답 모델 정의 ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    retrieved_document: Optional[str]
    question: str
    answer: str
    score: Optional[float]

# --- RAG 함수 정의 ---
def get_rag_response(question: str) -> dict:
    if not vector_store or not llm_pipeline:
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.")

    # 1. 관련 문서 검색 (유사도 점수 포함)
    results = vector_store.similarity_search_with_score(question, k=1)
    
    if not results:
        return {
            "retrieved_document": None,
            "answer": "관련 정보를 찾을 수 없습니다.",
            "score": None,
        }

    retrieved_doc, score = results[0]
    retrieved_text = retrieved_doc.page_content
    
    # 2. 질의 범위 제한 메커니즘
    score_threshold = 0.8
    if score < score_threshold:
        return {
            "retrieved_document": retrieved_text,
            "answer": "질문과 관련된 명확한 정보를 찾지 못했습니다. 위키피디아 문서의 일부 내용은 다음과 같습니다.",
            "score": float(score),
        }

    # 3. 프롬프트 생성
    prompt_template = f"""
    당신은 주어진 정보를 바탕으로 질문에 답변하는 AI 어시스턴트입니다. 반드시 아래 '문서 내용'만을 참고하여 답변을 생성해야 합니다. 문서에 없는 내용은 답변하지 마세요.

    [문서 내용]
    {retrieved_text}

    [질문]
    {question}

    [답변]
    """
    
    # 4. LLM을 통한 답변 생성
    generated_text = llm_pipeline(prompt_template)[0]['generated_text']
    
    # 5. 후처리: 프롬프트 부분을 제거하고 답변만 추출
    answer = generated_text.split("[답변]")[1].strip()
    
    return {
        "retrieved_document": retrieved_text,
        "answer": answer,
        "score": float(score),
    }

# --- API 엔드포인트 정의 ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        response_data = get_rag_response(request.question)
        return AnswerResponse(
            retrieved_document=response_data["retrieved_document"],
            question=request.question,
            answer=response_data["answer"],
            score=response_data["score"]
        )
    except Exception as e:
        print(f"'/ask' 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/")
def read_root():
    return {"message": "KorQuAD RAG API (Pinecone + KoBERT) 서버가 실행 중입니다. /docs 에서 API 문서를 확인하세요."}

