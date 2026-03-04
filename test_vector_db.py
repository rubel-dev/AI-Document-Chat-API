import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

chunks = [
    'Machine learning is a subset of AI.',
    "Deep learning uses neural networks.",
    'Biryani is a popular rice dish.'

]

embeddings = model.encode(chunks).tolist()

client = chromadb.Client()
collection = client.create_collection(name = 'docs')

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids = ['1', '2','3']
)

query = 'what is machine learing?'
q_emb = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[q_emb],
    n_results=2
)
print(results['documents'])