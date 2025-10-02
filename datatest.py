from datasets import load_dataset
dataset = load_dataset("squad_kor_v1", split="train")

# 첫 5개 샘플 출력
for i in range(5):
    print(f"\n=== 샘플 {i} ===")
    print(f"ID: {dataset[i]['id']}")
    print(f"Title: {dataset[i]['title']}")
    print(f"Context 길이: {len(dataset[i]['context'])} chars")
    print(f"Question: {dataset[i]['question']}")
    print(f"Answers: {dataset[i]['answers']}")