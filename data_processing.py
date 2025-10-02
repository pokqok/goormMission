import os
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from pinecone import Pinecone as PineconeClient, ServerlessSpec

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# --- 환경 변수 로드 ---
load_dotenv()

# --- 상수 정의 (BGE-M3 최적화) ---
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "korquad-rag-index")
EMBEDDING_DIMENSION = 1024

# BGE-M3에 최적화된 청킹 설정
CHUNK_SIZE = 2000  # 긴 context 보존
CHUNK_OVERLAP = 400

def recreate_pinecone_index():
    """Pinecone 인덱스를 1024 차원으로 재생성합니다."""
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    
    # 기존 인덱스 삭제
    if PINECONE_INDEX_NAME in pc.list_indexes().names():
        print(f"기존 '{PINECONE_INDEX_NAME}' 인덱스를 삭제합니다...")
        pc.delete_index(PINECONE_INDEX_NAME)
        print("삭제 완료.")
    
    # 새 인덱스 생성 (1024 차원)
    print(f"새 인덱스를 생성합니다 (dimension={EMBEDDING_DIMENSION})...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("인덱스 생성 완료!")
    return pc

def preprocess_and_embed_data():
    """KorQuAD 데이터를 BGE-M3로 임베딩하여 Pinecone에 저장합니다."""
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    # 1. Pinecone 인덱스 재생성
    pc = recreate_pinecone_index()
    
    # 인덱스 준비 대기
    import time
    print("인덱스 초기화 대기 중...")
    time.sleep(10)
    
    # 인덱스 확인
    index_description = pc.describe_index(PINECONE_INDEX_NAME)
    print(f"인덱스 차원: {index_description.dimension}")
    if index_description.dimension != EMBEDDING_DIMENSION:
        raise ValueError(f"인덱스 차원 불일치: {index_description.dimension} != {EMBEDDING_DIMENSION}")
    
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # 2. KorQuAD 데이터셋 로드
    print("\nKorQuAD v1 데이터셋을 로드합니다...")
    dataset = load_dataset("squad_kor_v1", split="train")
    
    print(f"데이터셋 로드 완료. 총 항목 수: {len(dataset)}")
    print("\n데이터 샘플:")
    print(f"  ID: {dataset[0]['id']}")
    print(f"  Title: {dataset[0]['title']}")
    print(f"  Question: {dataset[0]['question']}")
    print(f"  Context: {dataset[0]['context'][:100]}...")
    print(f"  Answers: {dataset[0]['answers']}\n")

    # 3. context별로 그룹화 (중복 제거)
    print("데이터를 context별로 그룹화합니다...")
    context_groups = {}
    for item in dataset:
        context = item['context']
        if context not in context_groups:
            context_groups[context] = []
        context_groups[context].append(item)
    
    print(f"고유 context 개수: {len(context_groups)}")
    print(f"전체 항목 수: {len(dataset)}")
    print("중복 제거 완료!\n")

    # 4. 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "다. ", ". ", "! ", "? ", "、", " ", ""],
        keep_separator=True
    )

    # 5. BGE-M3 임베딩 모델 로드
    print(f"BGE-M3 임베딩 모델을 로드합니다...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs={
            'device': 'cuda',
            'trust_remote_code': True
        },
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 16
        }
    )
    print("모델 로드 완료")

    # 6. 문서 처리 및 Pinecone 업로드
    print("\n문서를 청크로 분할하고 임베딩을 생성합니다...")
    batch_size = 50
    vectors_to_upsert = []
    total_chunks = 0
    
    for context_idx, (context, items) in enumerate(tqdm(context_groups.items(), desc="문서 처리")):
        # 대표 항목
        first_item = items[0]
        title = first_item['title']
        
        # 모든 관련 ID와 질문 수집
        all_ids = [item['id'] for item in items]
        all_questions = [item['question'] for item in items]
        
        # context를 청크로 분할
        chunks = text_splitter.split_text(context)
        
        chunk_batch = []
        chunk_metadata_batch = []
        
        for chunk_idx, chunk_text in enumerate(chunks):
            # 너무 짧은 청크 제외
            if len(chunk_text.strip()) < 100:
                continue
            
            # 고유한 청크 ID 생성
            chunk_id = f'ctx_{context_idx}_chunk_{chunk_idx}'
            
            chunk_batch.append(chunk_text)
            chunk_metadata_batch.append({
                'id': chunk_id,
                'original_id': all_ids[0],
                'all_ids': ','.join(all_ids[:10]),
                'chunk_idx': chunk_idx,
                'title': title,
                'text': chunk_text,
                'full_context': context,
                'sample_question': all_questions[0],
                'sample_answers': str(items[0]['answers']['text'][0]) if items[0]['answers']['text'] else '',
                'chunk_length': len(chunk_text),
                'num_related_questions': len(items),
                'model': 'bge-m3'
            })
        
        if not chunk_batch:
            continue
        
        # 배치 임베딩
        try:
            embeddings = embedding_model.embed_documents(chunk_batch)
            
            for embedding, metadata in zip(embeddings, chunk_metadata_batch):
                vectors_to_upsert.append({
                    'id': metadata['id'],
                    'values': embedding,
                    'metadata': metadata
                })
                total_chunks += 1
                
                if len(vectors_to_upsert) >= batch_size:
                    pinecone_index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
        
        except Exception as e:
            print(f"\n[ERROR] context {context_idx} 처리 중 오류: {e}")
            continue

    # 남은 벡터 업로드
    if vectors_to_upsert:
        pinecone_index.upsert(vectors=vectors_to_upsert)
    
    print(f"\n완료! 총 {total_chunks}개 청크가 Pinecone에 저장되었습니다.")
    
    # 인덱스 통계 확인
    time.sleep(5)
    stats = pinecone_index.describe_index_stats()
    print(f"Pinecone 인덱스 통계: {stats}")


def verify_index_quality(sample_size=5):
    """BGE-M3 인덱스 품질 검증"""
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    
    print("\n" + "="*50)
    print("BGE-M3 인덱스 품질 검증")
    print("="*50)
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs={
            'device': 'cuda',
            'trust_remote_code': True
        },
        encode_kwargs={
            'normalize_embeddings': True
        }
    )
    
    test_queries = [
        "유엔이 설립된 연도는?",
        "세종대왕의 업적은 무엇인가?",
        "한국의 수도는 어디인가?",
    ]
    
    for query in test_queries:
        print(f"\n질문: {query}")
        query_embedding = embedding_model.embed_query(query)
        
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
        if results['matches']:
            for i, match in enumerate(results['matches'], 1):
                score = match['score']
                metadata = match['metadata']
                
                print(f"  [{i}] Score: {score:.4f}")
                print(f"      Original ID: {metadata.get('original_id', 'N/A')}")
                print(f"      Title: {metadata.get('title', 'N/A')}")
                print(f"      Text: {metadata.get('text', '')[:150]}...")
        else:
            print("  검색 결과 없음")


if __name__ == "__main__":
    print("="*50)
    print("BGE-M3 임베딩 파이프라인 시작")
    print("="*50)
    print(f"모델: {EMBEDDING_MODEL_ID}")
    print(f"차원: {EMBEDDING_DIMENSION}")
    print(f"청크 크기: {CHUNK_SIZE}")
    print(f"오버랩: {CHUNK_OVERLAP}")
    print("="*50 + "\n")
    
    preprocess_and_embed_data()
    
    verify = input("\n인덱스 품질을 검증하시겠습니까? (y/n): ")
    if verify.lower() == 'y':
        verify_index_quality()