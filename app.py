import os
import re
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Langchain and Huggingface imports
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM
from pinecone import Pinecone as PineconeClient

# --- 환경 변수 로드 및 검증 ---
load_dotenv()
print("--- [app.py] 환경 변수 검증 시작 ---")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN") # 허깅페이스 토큰 로드
if PINECONE_API_KEY:
    print(f"PINECONE_API_KEY를 성공적으로 로드했습니다. (시작 부분: {PINECONE_API_KEY[:10]}...)")
else:
    print("경고: PINECONE_API_KEY를 찾을 수 없습니다.")
if HUGGING_FACE_TOKEN:
    print("HUGGING_FACE_TOKEN을 성공적으로 로드했습니다.")
else:
    print("경고: HUGGING_FACE_TOKEN을 찾을 수 없습니다. Gated model 접근에 실패할 수 있습니다.")
print("--- [app.py] 환경 변수 검증 종료 ---\n")

# --- 상수 정의 ---
EMBEDDING_MODEL_ID = "jhgan/ko-sbert-sts"
LLM_MODEL_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B" # LLM 모델 변경
PINECONE_INDEX_NAME = "korquad-rag-index"

# --- 전역 변수 ---
retriever = None
model_obj = None
tokenizer_obj = None

# 최대 사용할 컨텍스트 길이(문자).
MAX_CONTEXT_CHARS = 2000

# --- 모델 로드 ---
def load_models():
    """애플리케이션 시작 시 모든 모델을 로드합니다."""
    global retriever, model_obj, tokenizer_obj

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY 환경 변수가 설정되지 않아 Pinecone에 연결할 수 없습니다.")
    
    if not HUGGING_FACE_TOKEN:
        raise RuntimeError("HUGGING_FACE_TOKEN 환경 변수가 설정되지 않았습니다. Gated model에 접근할 수 없습니다.")

    # 1. 임베딩 모델 로드
    print(f"임베딩 모델을 로드합니다 ({EMBEDDING_MODEL_ID})...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)

    # 2. Pinecone 벡터 스토어 초기화
    print("Pinecone 벡터 스토어를 초기화합니다...")
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    vectorstore = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding_model
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

    # 3. LLM 로드 (HyperCLOVA X 모델에 맞게 수정)
    print(f"LLM을 로드합니다 ({LLM_MODEL_ID})...")
    
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True, 
        token=HUGGING_FACE_TOKEN 
    )
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_ID,
        token=HUGGING_FACE_TOKEN
    )

    # 전역에 저장
    model_obj = model
    tokenizer_obj = tokenizer
    
    print("모든 모델 로드가 완료되었습니다.")

# --- FastAPI 애플리케이션 설정 ---
app = FastAPI()

@app.on_event("startup")
def startup_event():
    load_models()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    retrieved_document_id: int
    retrieved_document: str
    question: str
    answers: str

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    global retriever, model_obj, tokenizer_obj

    if retriever is None or model_obj is None or tokenizer_obj is None:
        raise HTTPException(status_code=503, detail="모델이 아직 준비되지 않았습니다.")
    if not request.question:
        raise HTTPException(status_code=400, detail="질문이 비어있습니다.")

    try:
        # 1) Retriever로 문서 검색
        retrieved_docs = retriever.invoke(request.question)
        context_str = retrieved_docs[0].page_content if retrieved_docs else ""

        if len(context_str) > MAX_CONTEXT_CHARS:
            context_str = context_str[:MAX_CONTEXT_CHARS] + "\n... (중략)"

        # 2) 모델의 역할을 정보 '추출'로 한정하는 강력한 프롬프트 (수정)
        prompt_template = (
            "### 지시:\n"
            "당신은 주어진 '컨텍스트'에서 '질문'에 대한 정답을 찾는 정보 추출 AI입니다. 당신의 유일한 임무는 다음 규칙을 엄격하게 따르는 것입니다.\n\n"
            "### 규칙:\n"
            "1. '컨텍스트'를 분석하여 '질문'에 대한 답변을 찾으세요.\n"
            "2. '컨텍스트'에 답변에 해당하는 문장이 명확하게 있으면, 해당 문장을 바탕으로 간결한 한국어 답변을 생성하세요.\n"
            "3. '컨텍스트'에 답변에 해당하는 내용이 조금이라도 없다면, 절대로 외부 지식을 사용하거나 추측하지 말고, **오직** \"제공된 문서에서 관련 정보를 찾을 수 없습니다.\"라고만 출력해야 합니다.\n\n"
            "### 컨텍스트:\n{context}\n\n"
            "### 질문:\n{question}\n\n"
            "### 답변:\n"
        )
        prompt_text = prompt_template.format(context=context_str, question=request.question)

        # 3) 토크나이저로 입력 준비
        tokenizer = tokenizer_obj
        model = model_obj
        device = next(model.parameters()).device

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        # 4) 모델로 답변 생성 (결정적 모드, repetition_penalty 추가)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # 5) 생성된 부분만 디코딩
        input_length = inputs['input_ids'].shape[1]
        new_tokens = output_ids[0, input_length:]
        answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # 6) 최종 결과 반환
        return {
            "retrieved_document_id": 1,
            "retrieved_document": context_str,
            "question": request.question,
            "answers": answer_text or "모델이 답변을 생성하지 못했습니다."
        }

    except Exception as e:
        print(f"Exception in /ask: {e}")
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류 발생: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "KorQuAD RAG API가 실행 중입니다. /ask 엔드포인트로 POST 요청을 보내세요."}

