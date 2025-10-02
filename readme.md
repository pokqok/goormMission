# KorQuAD RAG 질의응답 시스템

위키피디아 데이터(KorQuAD 1.0)를 활용한 RAG 기반 질의응답 LLM 서비스

##  프로젝트 개요

KorQuAD 1.0 데이터셋을 기반으로 RAG(Retrieval-Augmented Generation) 시스템을 구축하여, 사용자의 질문에 대해 관련 위키피디아 문서를 검색하고 정확한 답변을 생성하는 REST API 서비스입니다.

### 주요 기능

- ✅ 질문 입력 시 관련 위키피디아 문서 검색
- ✅ 검색된 문서 기반 응답 생성
- ✅ 참조한 위키피디아 문서 출처 제공
- ✅ 질의 범위 제한 메커니즘 구현

##  시스템 아키텍처

```
사용자 질문
    ↓
[임베딩] BGE-M3 (1024차원)
    ↓
[벡터 검색] Pinecone (상위 20개)
    ↓
[재순위화] Cohere Rerank (상위 5개)
    ↓
[질의 범위 제한]
 • 관련성 임계값: 0.35
 • 평균 점수 검증
 • 점수 격차 분석
    ↓
[컨텍스트 구성] 점수별 가중치 (최대 3500자)
    ↓
[답변 생성] HyperCLOVAX LLM
    ↓
최종 답변 반환
```

##  기술 스택

| 구분 | 기술 |
|------|------|
| **언어** | Python 3.10+ |
| **웹 프레임워크** | FastAPI |
| **임베딩 모델** | BAAI/bge-m3 |
| **LLM** | naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B |
| **벡터 DB** | Pinecone Cloud |
| **Reranker** | Cohere rerank-multilingual-v3.0 |
| **주요 라이브러리** | LangChain, Transformers, HuggingFace Datasets |

##  설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd <repository-name>

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일 생성:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=korquad-rag-index
HUGGING_FACE_TOKEN=your_huggingface_token
COHERE_API_KEY=your_cohere_api_key
```

### 3. 데이터 전처리 및 인덱싱

```bash
python data_processing.py
```

- KorQuAD 1.0 데이터셋 다운로드
- 문단 단위 청크 분할 (overlap 포함)
- BGE-M3 임베딩 생성 (1024차원)
- Pinecone 인덱스에 저장

### 4. 서버 실행

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

서버 실행 후 `http://localhost:8000` 접속

##  API 사용 예시

### 질의응답 API

**Endpoint**: `POST /ask`

#### 요청 예시

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "유엔이 설립된 연도는?"}'
```

#### 응답 예시

```json
{
  "retrieved_document_id": "656656-0-0",
  "retrieved_document": "유엔(UN)은 국제 연합(United Nations)의 약자로, 1945년 10월 24일에 설립된 국제 기구이다...",
  "question": "유엔이 설립된 연도는?",
  "answers": "1945년입니다."
}
```

### 기타 엔드포인트

- `GET /`: API 정보 확인
- `GET /health`: 서버 상태 체크

##  구현 세부사항

### 1. 데이터 처리 (`data_processing.py`)

```python
# KorQuAD 1.0 데이터셋 로드
from datasets import load_dataset
dataset = load_dataset("squad_kor_v1")

# 청크 전략: 문단 단위 분할
# 임베딩: BGE-M3 (1024차원)
# 벡터 저장: Pinecone 클라우드
```

### 2. 검색 파이프라인 (`app.py`)

#### 단계별 프로세스

**Step 1: 초기 검색 (Pinecone)**
- 쿼리 임베딩 생성
- 유사도 기반 상위 20개 문서 검색

**Step 2: 재순위화 (Cohere Rerank)**
- 질문-문서 관련성 재평가
- 상위 5개 문서 선택

**Step 3: 질의 범위 제한**
```python
# 1. 최고 점수 체크
if best_score < 0.35:
    return "질문과 관련된 문서를 찾을 수 없습니다."

# 2. 평균 점수 검증 (최고 점수가 낮을 때)
if best_score < 0.5 and avg_score < 0.25:
    return "질문과 관련된 문서를 찾을 수 없습니다."

# 3. 점수 격차 분석
if score_gap > 0.7 and best_score < 0.5:
    return "질문과 관련된 문서를 찾을 수 없습니다."
```

**Step 4: 컨텍스트 구성**
- 점수별 가중치: >0.8 → 1500자, >0.6 → 1000자, 나머지 → 500자
- 전체 컨텍스트 최대 3500자

**Step 5: 답변 생성**
- 프롬프트: 문서 기반 답변 강제, 외부 지식 사용 금지
- 생성 파라미터: temperature=0.01, repetition_penalty=1.3

### 3. 프롬프트 엔지니어링

```
You are an AI that answers based solely on the given documents.

### Critical Rules:
1. Use ONLY the content from the [Document] below
2. If the information is not in the document, you MUST answer 
   "The information is not available in the document"
3. Do NOT guess or use external knowledge
4. Answer concisely in 1-2 sentences

### Examples:
[Few-shot 예시 포함]
```

##  모델 선택 과정 및 성능 개선

### 모델 변경 이력

1. **EleutherAI/polyglot-ko-1.3b** (초기)
   - 문제점: 답변이 짧으면 같은 말 반복, 길면 답변이 잘림

2. **kakaocorp/kanana-nano-2.1b-base** (1차 개선)
   - 성능 향상 확인
   - 예시:
     ```json
     {
       "question": "귀신고래의 서식지를 알려줘",
       "answers": "귀신토끼의 서식을 알려줌."  // 여전히 부정확
     }
     ```

3. **naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B** (최종 선택)
   - kanana보다 답변 제한에서 더 우수한 성능(문서에 없는 답변을 출력하지 않는 등)
   - 문서 기반 답변 준수율 향상
   - 예시:
     ```json
     {
       "question": "대한민국의 수도는 어디인가요?",
       "answers": "대한민국은 수도가 서울입니다."  // 정확한 답변
     }
     ```

### 핵심 개선 사항

- ✅ HyperCLOVAX의 Instruct 튜닝으로 지시사항 준수율 향상
- ✅ 답변 길이 제어 개선 (반복/잘림 현상 해결)
- ✅ 질의 범위 제한 메커니즘과의 조화

## 🔧 주요 설정 파라미터

```python
# 검색 설정
TOP_K_RETRIEVAL = 20        # Pinecone 초기 검색 개수
TOP_K_RERANK = 5            # Rerank 후 사용 개수
RELEVANCE_THRESHOLD = 0.35  # 최소 관련성 점수

# LLM 생성 설정
max_new_tokens = 120
temperature = 0.01
repetition_penalty = 1.3
```

##  접근 방식 및 특징

### 1. Rerank 기반 정확도 향상
- Cohere의 multilingual rerank 모델로 한국어 질문-문서 매칭 정확도 개선

### 2. 다단계 질의 범위 제한
- 임계값 + 평균 점수 + 격차 분석의 3중 검증
- Hallucination 최소화

### 3. 점수 기반 컨텍스트 가중치
- 관련성 높은 문서에 더 많은 토큰 할당
- 효율적인 컨텍스트 활용

### 4. 엄격한 프롬프트 제어
- Few-shot 예시로 문서 기반 답변 유도
- 외부 지식 사용 명시적 금지

## ⚙️ 시스템 요구사항

- **GPU**: CUDA 지원 필수
- **메모리**: 
  - 임베딩 모델: ~2GB
  - LLM: ~3GB (FP16)
- **Python**: 3.10 이상

##  프로젝트 구조

```
.
├── app.py                 # FastAPI 서버 메인
├── data_processing.py     # 데이터 전처리 및 인덱싱
├── requirements.txt       # 패키지 의존성
├── .env                   # 환경 변수 (git 제외)
└── README.md             # 프로젝트 문서
```

##  라이선스

본 프로젝트는 교육 목적으로 제작되었습니다.


---

**Built with using KorQuAD, Pinecone, Cohere, and HuggingFace**