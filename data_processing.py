import os
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv
from tqdm.auto import tqdm

# --- 환경 변수 로드 및 검증 ---
load_dotenv()
print("--- [data_processing.py] 환경 변수 검증 시작 ---")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY:
    print(f"PINECONE_API_KEY를 성공적으로 로드했습니다. (시작 부분: {PINECONE_API_KEY[:10]}...)")
else:
    print("경고: PINECONE_API_KEY를 찾을 수 없습니다.")
print("--- [data_processing.py] 환경 변수 검증 종료 ---\n")

# --- 상수 정의 ---
EMBEDDING_MODEL_ID = "jhgan/ko-sbert-sts"
PINECONE_INDEX_NAME = "korquad-rag-index"
MODEL_DIMENSION = 768 # jhgan/ko-sbert-sts 모델의 임베딩 차원은 768입니다.

def preprocess_and_embed_data():
    """
    KorQuAD 데이터셋을 로드, 전처리, 임베딩하여 Pinecone에 저장합니다.
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

    # 1. Pinecone 클라이언트 초기화 및 인덱스 검증
    print("Pinecone 클라이언트를 초기화합니다...")
    pc = PineconeClient(api_key=PINECONE_API_KEY)

    print(f"'{PINECONE_INDEX_NAME}' 인덱스를 확인합니다...")
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"Pinecone에 '{PINECONE_INDEX_NAME}' 인덱스가 존재하지 않습니다. 먼저 Pinecone 웹사이트에서 인덱스를 생성해주세요.")

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    index_stats = pinecone_index.describe_index_stats()
    index_dimension = index_stats.get('dimension')

    if index_dimension != MODEL_DIMENSION:
        raise ValueError(
            f"Pinecone 인덱스의 차원({index_dimension})이 모델의 차원({MODEL_DIMENSION})과 일치하지 않습니다.\n"
            f"Pinecone에서 인덱스를 삭제하고, Dimension을 {MODEL_DIMENSION}으로 설정하여 다시 생성해주세요."
        )
    print(f"인덱스 차원이 ({index_dimension}) 모델 차원과 일치함을 확인했습니다.")


    # 2. KorQuAD 데이터셋 로드
    print("KorQuAD v1 데이터셋을 로드합니다...")
    dataset = load_dataset("squad_kor_v1", split="train")

    # context와 title을 기준으로 고유 문서 추출
    unique_contexts = {}
    for record in tqdm(dataset, desc="고유 문서 추출 중"):
        context = record['context']
        title = record['title']
        if context not in unique_contexts:
            unique_contexts[context] = {"title": title, "context": context}

    documents = list(unique_contexts.values())
    print(f"데이터셋 로드 완료. 고유 문서 개수: {len(documents)}")


    # 3. 텍스트 분할
    print("문서를 청크(chunk) 단위로 분할합니다...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for i, doc in enumerate(documents):
        # Langchain이 처리할 수 있는 Document 형태로 변환
        from langchain.docstore.document import Document
        temp_doc = Document(page_content=doc['context'], metadata={"title": doc['title'], "doc_id": f"doc_{i}"})
        split_docs = text_splitter.split_documents([temp_doc])
        chunks.extend(split_docs)
    
    print(f"분할된 청크 개수: {len(chunks)}")


    # 4. 임베딩 모델 로드
    print(f"임베딩 모델을 로드합니다 ({EMBEDDING_MODEL_ID})...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID)


    # 5. 임베딩 생성 및 Pinecone에 저장
    print("임베딩을 생성하고 Pinecone에 저장합니다...")
    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="Pinecone 업로드 중"):
        batch_chunks = chunks[i:i + batch_size]
        
        texts_to_embed = [chunk.page_content for chunk in batch_chunks]
        embeddings = embedding_model.embed_documents(texts_to_embed)
        
        vectors_to_upsert = []
        for j, chunk in enumerate(batch_chunks):
            vector = {
                "id": f"{chunk.metadata['doc_id']}_chunk_{i+j}",
                "values": embeddings[j],
                "metadata": {
                    "text": chunk.page_content,
                    "title": chunk.metadata['title']
                }
            }
            vectors_to_upsert.append(vector)
            
        pinecone_index.upsert(vectors=vectors_to_upsert)

    print(f"{len(chunks)}개 청크 처리 완료")
    print("Pinecone 벡터 스토어 저장이 완료되었습니다.")


if __name__ == "__main__":
    preprocess_and_embed_data()

