# KorQuAD RAG 질의응답 시스템

위키피디아 데이터(KorQuAD 1.0)를 활용한 RAG 기반 질의응답 LLM 서비스

## 📋 프로젝트 개요

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

## 🛠️ 기술 스택

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
git clone https://github.com/pokqok/goormMission.git
cd goormMission

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. PyTorch CUDA 설치 (필수)

⚠️ **중요**: requirements.txt 설치 전에 PyTorch CUDA 버전을 먼저 설치해야 합니다.

```bash
# CUDA 12.6 버전 PyTorch 설치
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env` 파일 생성:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=korquad-rag-index
HUGGING_FACE_TOKEN=your_huggingface_token
COHERE_API_KEY=your_cohere_api_key
```

### 5. 데이터 전처리 및 인덱싱

```bash
python data_processing.py
```

- KorQuAD 1.0 데이터셋 다운로드
- 문단 단위 청크 분할 (overlap 포함)
- BGE-M3 임베딩 생성 (1024차원)
- Pinecone 인덱스에 저장

### 6. 서버 실행

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

# 청크 전략: RecursiveCharacterTextSplitter 사용 (크기: 2000, 중첩: 400)
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

## 개발 과정 및 트러블슈팅

### 주요 도전과제와 해결 과정

#### 1. LLM 모델 선택의 어려움

**문제 1: EleutherAI/polyglot-ko-1.3b**
- 답변이 짧을 때: 같은 내용을 반복 생성
- 답변이 길 때: 문장이 중간에 잘리는 현상
- 원인: 모델 크기(1.3B)의 한계와 Instruction tuning 부족

**문제 2: kakaocorp/kanana-nano-2.1b-base**
- 성능은 향상되었으나 여전히 부정확한 답변 생성
- 실제 사례:
  ```
  질문: "귀신고래의 서식지를 알려줘"
  답변: "귀신토끼의 서식을 알려줌"  // 엉뚱한 답변
  ```
- 문제: 문서 내용 무시하고 유사 단어로 답변 생성

**최종 해결: naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B**
- Instruct 튜닝으로 지시사항 준수율 대폭 향상
- 문서 기반 답변 강제 프롬프트에 잘 반응
- 실제 개선 사례:
  ```
  질문: "대한민국의 수도는 어디인가요?"
  문서: 임시정부 관련 내용 (수도 언급 없음)
  이전: "대한민국의 수도는 서울입니다" (Hallucination)
  개선: 문서 확인 후 적절한 답변 생성
  ```

#### 2. Hallucination 방지의 어려움

**문제 상황**
```python
질문: "핀란드 수도는?"
문서: 필리핀 세부 관련 내용
답변: "핀란드의 수도는 헬싱키입니다"  // 외부 지식 사용
```

**시도한 해결 방법들**

1. **프롬프트 강화 (1차 시도)**
   - 규칙을 명확히 명시
   - 결과: 여전히 외부 지식 사용

2. **후처리 검증 추가 (2차 시도)**
   - 숫자가 문서에 있는지 검증
   - 핵심 단어가 문서에 있는지 검증
   - 문제: 너무 엄격해서 정답도 차단
   ```
   질문: "주토피아의 주디홉스의 꿈은?"
   문서: "경찰을 꿈꾼다" (명시)
   LLM: "경찰이 되는 것입니다"
   검증: 실패 (단어 불일치로 오판)
   ```

3. **질의 범위 제한 메커니즘 (최종 해결)**
   - Cohere Rerank 점수 기반 3단계 검증
   - 임계값: 0.35 (실험을 통해 결정)
   - 평균 점수 체크: 상위 문서만 높고 나머지가 낮으면 차단
   - 점수 격차 분석: 1위와 2위 차이가 너무 크면 의심

#### 3. Rerank 최적화 과정

**초기 설정의 문제**
```
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 1
```
- 너무 적은 후보로 좋은 문서 놓침
- 1개만 사용하여 정보 부족

**최적화 과정**
```
1차: TOP_K_RETRIEVAL = 10, TOP_K_RERANK = 3
→ 여전히 후보 부족

2차: TOP_K_RETRIEVAL = 20, TOP_K_RERANK = 5
→ 정확도 향상

3차: 점수별 가중치 추가
- >0.8: 1500자
- >0.6: 1000자
- 나머지: 500자
→ 최종 채택
```

**실제 개선 효과**
```
질문: "주토피아는 어떤 영화인가요?"
이전: Score 0.04 (관련성 낮음) → 검색 실패
이후: Score 0.99 (1위) → 정확한 답변
```

#### 4. 프롬프트 엔지니어링 시행착오

**시도 1: 한국어 프롬프트**
- 문제: 영어로 답변하거나 혼란스러운 출력
- HyperCLOVAX가 영어 지시사항에 더 잘 반응

**시도 2: 영어 프롬프트 + CoT**
```
Thinking: ...
Answer: ...
```
- 문제: 생성 속도 저하, 불필요한 Thinking 부분 생성

**시도 3: 영어 프롬프트 + Step-by-Step**
- 문제: 여전히 복잡함

**최종: 간결한 영어 프롬프트 + Few-shot 예시**
- Critical Rules 4가지로 간소화
- 긍정/부정 예시 명확히 제시
- 결과: 가장 안정적인 성능

#### 5. 환경 설정 문제

**문제 1: PyTorch CPU 버전 설치**
```
ERROR: AssertionError: Torch not compiled with CUDA enabled
```
- 원인: pip install torch 시 자동으로 CPU 버전 설치
- 해결: CUDA 버전 명시적 설치
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

**문제 2: Pinecone 버전 충돌**
```
AttributeError: module 'pinecone' has no attribute 'Index'
```
- 원인: pinecone==7.3.0과 langchain_community 호환성
- 해결: langchain_pinecone의 PineconeVectorStore 사용
```python
# 이전 (오류)
from langchain_community.vectorstores import Pinecone
vectorstore = Pinecone.from_existing_index(...)

# 수정 (정상)
from langchain_pinecone import PineconeVectorStore
index = pc.Index(index_name)
vectorstore = PineconeVectorStore(index=index, ...)
```

**문제 3: requirements.txt 재현성**
- 버전 미지정으로 다른 환경에서 설치 실패
- 해결: 주요 패키지 버전 명시, PyTorch는 별도 설치 안내

#### 6. 데이터 전처리 최적화

**청크 전략 결정**
- RecursiveCharacterTextSplitter 사용
- chunk_size: 2000 (실험: 1000/1500/2000)
- chunk_overlap: 400 (20% 중첩)
- 이유: 문맥 유지하면서 임베딩 품질 최대화

### 학습한 교훈

1. **작은 모델의 한계**: 1.5B 이하 모델은 복잡한 지시사항 준수 어려움
2. **프롬프트의 중요성**: 복잡한 후처리보다 명확한 프롬프트가 더 효과적
3. **Rerank의 효과**: 초기 검색보다 재순위화가 정확도에 더 큰 영향
4. **환경 재현성**: 명시적 버전 지정과 설치 순서가 중요
5. **검증의 균형**: 너무 엄격한 검증은 오히려 정답을 차단할 수 있음

## 모델 선택 과정 및 성능 개선

### 모델 변경 이력

**제약 조건**
- 시간 제약: 제한된 개발 기간으로 충분한 정량/정성 평가 불가
- 모델 크기: GPU 메모리 제한으로 2B 이내 모델로 제한
- 검색 범위: 한국어 지원 모델 위주로 제한적 탐색

1. **EleutherAI/polyglot-ko-1.3b** (초기)
   - 선정 이유: 대표적인 한국어 오픈소스 모델
   - 문제점: 답변이 짧으면 같은 말 반복, 길면 답변이 잘림
   - 평가: 제한적 테스트로 초기 발견

2. **kakaocorp/kanana-nano-2.1b-base** (1차 개선)
   - 선정 이유: Kakao에서 공개한 경량 한국어 모델
   - 성능 향상 확인
   - 예시:
     ```json
     {
       "question": "귀신고래의 서식지를 알려줘",
       "answers": "귀신토끼의 서식을 알려줌."  // 여전히 부정확
     }
     ```
   - 한계: 문서 기반 답변 준수율 낮음

3. **naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B** (최종 선택)
   - 선정 이유: Naver의 Instruct 튜닝 모델 발견
   - kanana보다 답변 제한에서 더 우수한 성능(문서에 없는 답변을 출력하지 않는 등)
   - 문서 기반 답변 준수율 향상
   - 예시:
     ```json
     {
       "question": "대한민국의 수도는 어디인가요?",
       "answers": "대한민국은 수도가 서울입니다."  // 정확한 답변
     }
     ```

### 프로젝트의 한계 및 개선 방향

**한계점**

1. **모델 평가의 부족**
   - 정량 평가: KorQuAD 테스트셋 기반 EM/F1 스코어 미측정
   - 정성 평가: 다양한 질문 유형별 체계적 테스트 미실시
   - 비교 실험: 3개 모델 외 추가 모델 비교 부족
   - 원인: 제한된 개발 기간

2. **모델 선정 범위의 제약**
   - 탐색 범위: HuggingFace에서 "korean", "ko", "2B 이하" 키워드로 제한적 검색
   - 미탐색 모델: KULLM, KoAlpaca 등 다른 한국어 모델 미평가
   - 크기 제약: GPU 메모리(약 8GB 여유)로 인해 2B 초과 모델 제외
   - 검색 시간: 각 모델당 1-2시간 테스트로 신속 결정

3. **성능 최적화 여지**
   - 프롬프트: Few-shot 예시 개수/내용 최적화 미실시
   - Rerank 파라미터: TOP_K, 임계값 등 Grid Search 미실시
   - 청크 전략: 다양한 크기/overlap 조합 체계적 실험 부족

**향후 개선 방향**

1. **체계적 모델 평가**
   - KorQuAD 1.0 검증셋 기반 정량 평가
   - Hallucination rate 측정
   - 답변 품질 인간 평가

2. **LLM API 활용**
   - OpenAI GPT-4, Anthropic Claude 등 상용 API 적용
   - 한국어 특화 API (Clova X, HyperCLOVA X) 테스트
   - API 비용 대비 성능 개선 효과 분석
   - Latency와 품질 트레이드오프 평가

3. **하이퍼파라미터 최적화**
   - Rerank TOP_K: 3/5/7/10 비교
   - 임계값: 0.3/0.35/0.4 실험
   - 청크 크기: 1500/2000/2500 비교

4. **시스템 안정성 향상**
   - 에러 처리 강화
   - 캐싱 메커니즘 도입
   - 배치 처리 최적화

### 핵심 개선 사항

- ✅ HyperCLOVAX의 Instruct 튜닝으로 지시사항 준수율 향상
- ✅ 답변 길이 제어 개선 (반복/잘림 현상 해결)
- ✅ 질의 범위 제한 메커니즘과의 조화

##  주요 설정 파라미터

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

##  시스템 요구사항

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
