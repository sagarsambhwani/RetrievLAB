from retrievlab.embeddings.fastembed import FastEmbedClient
client = FastEmbedClient()

vectors = client.get_embeddings([
    "FastAPI",
    "Docker",
])

print(type(vectors))
print(type(vectors[0]))
print(len(vectors))
print(len(vectors[0]))