import os
import re
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
LLM_MODEL_ID = "kakaocorp/kanana-nano-2.1b-base" # LLM 모델 변경
PINECONE_INDEX_NAME = "korquad-rag-index"

# --- 전역 변수 및 프롬프트 템플릿 ---
retriever = None
pipe = None
model_obj = None
tokenizer_obj = None

# 프롬프트 수정: '한 문장' 제약을 '핵심 내용 요약'으로 변경
PROMPT_TEMPLATE = (
    "당신은 다음 주어진 컨텍스트만 을 사용해 질문에 답변하는 AI야.\n"
    "1) 컨텍스트 외 정보를 절대 추가하면 안되.\n"
    "2) 컨텍스트의 핵심 내용을 요약하여 한국어로 답변해.\n"
    "3) 만약 제공된 컨텍스트에서 답을 찾을 수 없으면 정확히 다음 문장만 출력해:\n"
    "   \"제공된 문서에서 답을 찾을 수 없습니다.\"\n\n"
    "컨텍스트:\n{context}\n\n"
    "질문: {question}\n\n"
    "답변:"
)

# 최대 사용할 컨텍스트 길이(문자). 너무 길면 모델 혼동하므로 자름.
MAX_CONTEXT_CHARS = 1600

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
    # 요구사항 반영: 가장 관련성 높은 1개의 문서만 검색하도록 수정
    retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

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

# 요구사항 반영: 응답 모델을 명세에 맞게 수정
class AnswerResponse(BaseModel):
    retrieved_document_id: int
    retrieved_document: str
    question: str
    answers: str

# --- ask_question 엔드포인트: model.generate 직접 호출 + 엄격한 프롬프트 사용 ---
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    global retriever, model_obj, tokenizer_obj

    if retriever is None or model_obj is None or tokenizer_obj is None:
        raise HTTPException(status_code=503, detail="모델/파이프라인이 아직 준비되지 않았습니다.")
    if not request.question:
        raise HTTPException(status_code=400, detail="질문이 비어있습니다.")

    try:
        # 1) Retriever로 문서 검색 (가장 관련성 높은 1개)
        retrieved_docs = retriever.invoke(request.question)
        
        context_str = ""
        if retrieved_docs:
            context_str = retrieved_docs[0].page_content

        # 2) 컨텍스트 길이 제한 (너무 길면 절단)
        if len(context_str) > MAX_CONTEXT_CHARS:
            context_str = context_str[:MAX_CONTEXT_CHARS].rsplit("\n", 1)[0] + "\n... (중략)"

        # 3) Prompt 생성 (엄격 지침 포함)
        prompt_text = PROMPT_TEMPLATE.format(context=context_str, question=request.question)
        
        # ---------------- 안전한 generate + 후처리 (기존 안정 버전) ----------------
        tokenizer = tokenizer_obj
        model = model_obj
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        
        eos_id = getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "sep_token_id", None) or getattr(tokenizer, "pad_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", eos_id)
        
        enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        gen_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
        
        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        
        if full_text.startswith(prompt_text):
            answer_text = full_text[len(prompt_text):].strip()
        else:
            fallback_trim = prompt_text[-200:]
            idx = full_text.find(fallback_trim)
            if idx != -1:
                answer_text = full_text[idx+len(fallback_trim):].strip()
            else:
                qidx = full_text.find(request.question)
                if qidx != -1:
                    answer_text = full_text[qidx + len(request.question):].strip()
                else:
                    answer_text = full_text.strip()
        
        answer_text = re.sub(r'\s+', ' ', answer_text).strip()
        
        def is_incomplete(s):
            return not bool(re.search(r'[\.다요\?！\!]\s*$', s)) and len(s) > 0
        
        retry_cnt = 0
        while is_incomplete(answer_text) and retry_cnt < 2:
            retry_cnt += 1
            cont_prompt = prompt_text + "\nCONTINUE: " + answer_text
            enc2 = tokenizer(cont_prompt, return_tensors="pt", truncation=True, max_length=4096)
            ids2 = enc2["input_ids"].to(device)
            att2 = enc2.get("attention_mask", None)
            if att2 is not None:
                att2 = att2.to(device)
            gen2 = model.generate(
                ids2,
                attention_mask=att2,
                max_new_tokens=64,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
            )
            full2 = tokenizer.decode(gen2[0], skip_special_tokens=True)
            if full2.startswith(cont_prompt):
                addition = full2[len(cont_prompt):].strip()
            else:
                pidx = full2.find(answer_text)
                if pidx != -1:
                    addition = full2[pidx + len(answer_text):].strip()
                else:
                    addition = full2.strip()
            answer_text = (answer_text + " " + addition).strip()
            answer_text = re.sub(r'\s+', ' ', answer_text).strip()

        # -------------------------------------------------------------------

        # 최종 결과 반환 (요구사항 형식에 맞춤)
        return {
            "retrieved_document_id": 1,
            "retrieved_document": context_str,
            "question": request.question,
            "answers": answer_text
        }

    except HTTPException:
        raise
    except Exception as e:
        print("Exception in /ask:", str(e))
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류 발생: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "KorQuAD RAG API가 실행 중입니다. /ask 엔드포인트로 POST 요청을 보내세요."}

