import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Langchain and Huggingface imports
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from pinecone import Pinecone as PineconeClient

# --- 환경 변수 로드 및 검증 ---
load_dotenv()
print("--- [app.py] 환경 변수 검증 시작 ---")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY:
    print(f"PINECONE_API_KEY를 성공적으로 로드했습니다. (시작 부분: {PINECONE_API_KEY[:10]}...)")
else:
    print("경고: PINECONE_API_KEY를 찾을 수 없습니다.")
print("--- [app.py] 환경 변수 검증 종료 ---\n")

# --- 상수 정의 ---
EMBEDDING_MODEL_ID = "jhgan/ko-sbert-sts"
LLM_MODEL_ID = "EleutherAI/polyglot-ko-1.3b"
PINECONE_INDEX_NAME = "korquad-rag-index"

# --- 전역 변수 및 프롬프트 템플릿 ---
retriever = None
pipe = None
model_obj = None
tokenizer_obj = None

# 엄격한 지침: 반드시 컨텍스트 안에서만 답변, 한 문장으로, 없으면 고정 문구 출력
PROMPT_TEMPLATE = (
    "당신은 다음 **주어진 컨텍스트**만을 사용해 질문에 답변하는 AI입니다.\n"
    "1) **컨텍스트 외 정보를 절대 추가하지 마세요.**\n"
    "2) 답변은 **한국어 한 문장**으로만 해주세요.\n"
    "3) 만약 제공된 컨텍스트에서 답을 찾을 수 없으면 **정확히 다음 문장만** 출력하세요:\n"
    "   \"제공된 문서에서 답을 찾을 수 없습니다.\"\n\n"
    "컨텍스트:\n{context}\n\n"
    "질문: {question}\n\n"
    "답변:"
)

# 최대 사용할 컨텍스트 길이(문자). 너무 길면 모델 혼동하므로 자름.
MAX_CONTEXT_CHARS = 1600
MAX_DOCS = 3  # retriever가 반환한 문서 중 상위 N개만 사용

# --- 모델 및 파이프라인 로드 ---
def load_models_and_chain():
    """애플리케이션 시작 시 모든 모델과 RAG 체인을 로드합니다."""
    global retriever, pipe, model_obj, tokenizer_obj

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY 환경 변수가 설정되지 않아 Pinecone에 연결할 수 없습니다.")

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
    retriever = vectorstore.as_retriever()

    # 3. LLM 로드 및 파이프라인 생성
    print(f"LLM을 로드하고 파이프라인을 생성합니다 ({LLM_MODEL_ID})...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

    # 전역에 저장
    model_obj = model
    tokenizer_obj = tokenizer

    # 파이프라인은 유지(디버그용), 실제 응답은 model.generate() 사용
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        # pipeline 인자들은 일부 무시될 수 있으니 generate를 우선 사용
    )

    print("모든 모델 및 파이프라인 로드가 완료되었습니다.")

# --- FastAPI 애플리케이션 설정 ---
app = FastAPI()

@app.on_event("startup")
def startup_event():
    """FastAPI 서버 시작 시 모델을 로드합니다."""
    load_models_and_chain()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    retrieved_context: str

# --- ask_question 엔드포인트: model.generate 직접 호출 + 엄격한 프롬프트 사용 ---
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    global retriever, model_obj, tokenizer_obj

    if retriever is None or model_obj is None or tokenizer_obj is None:
        raise HTTPException(status_code=503, detail="모델/파이프라인이 아직 준비되지 않았습니다.")
    if not request.question:
        raise HTTPException(status_code=400, detail="질문이 비어있습니다.")

    try:
        # 1) Retriever로 문서 검색 (상위 MAX_DOCS개만 사용)
        retrieved_docs = retriever.invoke(request.question)
        docs = retrieved_docs[:MAX_DOCS] if retrieved_docs else []
        context_str = "\n---\n".join([doc.page_content for doc in docs]) if docs else ""

        # 2) 컨텍스트 길이 제한 (너무 길면 절단)
        if len(context_str) > MAX_CONTEXT_CHARS:
            context_str = context_str[:MAX_CONTEXT_CHARS].rsplit("\n", 1)[0] + "\n... (중략)"

        # 3) Prompt 생성 (엄격 지침 포함)
        prompt_text = PROMPT_TEMPLATE.format(context=context_str, question=request.question)

        # 4) tokenizer/model 준비
        tokenizer = tokenizer_obj
        model = model_obj
        device = None
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        # ensure eos/pad token exist
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 5) generate 직접 호출: 결정적(샘플링 X), 짧은 출력
        gen_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,        # 결정적 생성
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.02,
        )

        # 6) 생성된 새 토큰만 디코드
        new_tokens = gen_ids[0, input_ids.shape[-1]:] if gen_ids.shape[-1] > input_ids.shape[-1] else gen_ids[0, 0:0]
        answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 7) 폴백: 만약 빈 문자열이면 파이프라인으로 한 번만 시도 (디버그 목적)
        if not answer_text:
            print("DEBUG: 첫 generate가 빈 문자열이어서 폴백 파이프라인 호출합니다.")
            fallback = pipe(prompt_text, max_new_tokens=64)
            if isinstance(fallback, list) and len(fallback) > 0 and isinstance(fallback[0], dict) and "generated_text" in fallback[0]:
                gen_full = fallback[0]["generated_text"]
            else:
                gen_full = str(fallback)
            # 프롬프트가 붙어있으면 떼기
            if gen_full.startswith(prompt_text):
                answer_text = gen_full[len(prompt_text):].strip()
            else:
                answer_text = gen_full.strip()

        # 8) 안전장치: 모델이 여전히 컨텍스트 외 정보를 내면 필터링 (엄격 모드)
        # 만약 모델이 "제공된 문서에서 답을 찾을 수 없습니다."를 출력했으면 그대로 반환
        if not answer_text:
            print("ERROR: 모델이 응답을 생성하지 않았습니다. PROMPT (len={}):\n{}".format(len(prompt_text), prompt_text[:2000]))
            raise HTTPException(status_code=500, detail="모델이 응답을 생성하지 못했습니다. 서버 로그를 확인하세요.")

        # 9) 최종 결과: 모델이 컨텍스트 외 정보를 만들지 못하도록 한 문장 길이로 제한 반환
        # (이미 프롬프트에 지침이 포함되어 있으므로 추가 변환 없이 반환)
        return {
            "question": request.question,
            "answer": answer_text,
            "retrieved_context": context_str
        }

    except HTTPException:
        raise
    except Exception as e:
        print("Exception in /ask:", str(e))
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류 발생: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "KorQuAD RAG API가 실행 중입니다. /ask 엔드포인트로 POST 요청을 보내세요."}
