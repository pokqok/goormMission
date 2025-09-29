import os
from datasets import load_dataset
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import pinecone
from pinecone import Pinecone as PineconeClient, PodSpec

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Pinecone 설정 ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "korquad-rag-index"

def preprocess_and_embed_data():
    """
    KorQuAD 데이터셋을 로드, 전처리하고 Pinecone 벡터 스토어에 임베딩하여 저장합니다.
    """
    if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
        raise ValueError("PINECONE_API_KEY와 PINECONE_ENVIRONMENT 환경 변수를 설정해야 합니다.")

    # 1. Pinecone 초기화 및 인덱스 생성
    print("Pinecone을 초기화하고 인덱스를 확인/생성합니다...")
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"'{INDEX_NAME}' 인덱스를 생성합니다. (차원: 768)")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,  # 'skt/kobert-base-v1' 모델의 임베딩 차원
            metric='cosine',
            spec=PodSpec(
                environment=PINECONE_ENVIRONMENT
            )
        )
        print("인덱스 생성 완료.")
    else:
        print(f"'{INDEX_NAME}' 인덱스가 이미 존재합니다.")

    # 2. 데이터셋 로드
    print("KorQuAD v1 데이터셋을 로드합니다...")
    dataset = load_dataset("skt/kobert-base-v1")
    corpus = [item['context'] for item in dataset['train']]
    unique_corpus = list(set(corpus))
    print(f"데이터셋 로드 완료. 고유 문서 개수: {len(unique_corpus)}")

    # 3. 텍스트 분할
    print("문서를 청크(chunk) 단위로 분할합니다...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.create_documents(unique_corpus)
    print(f"분할된 청크 개수: {len(split_docs)}")

    # 4. 임베딩 모델 로드 (KoBERT로 변경)
    print("임베딩 모델을 로드합니다 (skt/kobert-base-v1)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="skt/kobert-base-v1",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    # 5. Pinecone 벡터 스토어에 데이터 저장
    print("Pinecone 벡터 스토어에 데이터를 저장합니다. 다소 시간이 걸릴 수 있습니다...")
    try:
        Pinecone.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            index_name=INDEX_NAME
        )
        print("Pinecone 벡터 스토어 저장이 완료되었습니다.")
    except Exception as e:
        print(f"Pinecone에 데이터 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    preprocess_and_embed_data()

