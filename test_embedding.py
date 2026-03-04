from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [
    'what is machine learning?',
    'Explain Ml in simple words.',
    'How to cook biryani?'
]

emb = model.encode(texts)
print(emb.shape)
print("number of embeddings: ", len(emb))
print("Embedding dimension: ", len(emb[0]))

a, b, c = emb[0], emb[1], emb[2]
cos_ab = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
cos_ac = np.dot(a, c) / (np.linalg.norm(a)*np.linalg.norm(c))

print('sim(1, 2) = ', cos_ab)
print('sim(1, 3) = ', cos_ac)

query = 'what is machine learing?'
q_vec = model.encode(query)

for i, v in enumerate(emb):
    sim = np.dot(q_vec, v) /(np.linalg.norm(q_vec)) * np.linalg.norm(v)
    print(f"sim(query, text{i+1})=", sim)

