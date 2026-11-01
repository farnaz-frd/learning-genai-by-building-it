# rag_v2.py — production-grade RAG with real embeddings + ChromaDB
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
import time

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
MODEL  = "llama3.1:8b"

# ─────────────────────────────────────────
# STEP 1: load embedding model
# runs locally, no API key needed
# ─────────────────────────────────────────
print("loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ embedding model ready\n")

# ─────────────────────────────────────────
# STEP 2: knowledge base — same docs + more
# ─────────────────────────────────────────
DOCUMENTS = [
    {"id": "doc_0",  "text": "The Transformer architecture was introduced in 2017 in the paper 'Attention is All You Need' by Vaswani et al. It replaced recurrent networks with self-attention mechanisms, enabling parallel processing of sequences."},
    {"id": "doc_1",  "text": "Self-attention computes Query, Key, and Value matrices from the input. The attention score between tokens i and j is computed as Q_i dot K_j divided by sqrt(d_k), then softmax is applied to get weights for the Value vectors."},
    {"id": "doc_2",  "text": "LoRA (Low-Rank Adaptation) is a fine-tuning technique that freezes the original model weights and adds trainable low-rank matrices A and B. The weight update is expressed as delta_W = B times A, where the rank r is much smaller than the original dimensions."},
    {"id": "doc_3",  "text": "Large Language Models are trained on next-token prediction. Given a sequence of tokens, the model learns to predict the most probable next token. This simple objective forces the model to learn grammar, facts, reasoning and world knowledge."},
    {"id": "doc_4",  "text": "Overfitting occurs when training loss decreases but validation loss increases. The model memorizes training data instead of learning general patterns. Solutions include dropout, regularization, early stopping, and getting more training data."},
    {"id": "doc_5",  "text": "RAG (Retrieval-Augmented Generation) combines a retrieval system with a language model. Relevant documents are retrieved from a knowledge base and injected into the LLM prompt as context, reducing hallucination and keeping knowledge up to date."},
    {"id": "doc_6",  "text": "Multi-head attention runs h independent attention operations in parallel. Each head projects the input into a lower-dimensional subspace using separate Wq, Wk, Wv matrices. Heads learn to specialize in different relationship types through training."},
    {"id": "doc_7",  "text": "The causal mask in decoder-only transformers prevents each token from attending to future tokens. It is implemented as an upper-triangular matrix of negative infinity values added to attention scores before softmax."},
    {"id": "doc_8",  "text": "Positional encoding adds position information to token embeddings since attention has no inherent sense of order. Sinusoidal encodings use sine and cosine functions at different frequencies. RoPE encodes relative position by rotating query and key vectors."},
    {"id": "doc_9",  "text": "LayerNorm normalizes activations over the last dimension (per token, per position). Unlike BatchNorm it is independent of batch size and sequence length, making it suitable for variable-length sequences and batch size of 1 at inference."},
    {"id": "doc_10", "text": "QLoRA combines quantization with LoRA. The base model is quantized to 4-bit precision, reducing memory by 4x. LoRA adapters are then trained in 16-bit on top of the frozen quantized model. This enables fine-tuning LLaMA 70B on a single consumer GPU."},
    {"id": "doc_11", "text": "The feedforward network in each transformer block consists of two linear layers with a GELU activation between them. The intermediate dimension is typically 4x the model dimension. Research suggests FFN layers store factual knowledge learned during pretraining."},
    {"id": "doc_12", "text": "RLHF (Reinforcement Learning from Human Feedback) trains a reward model on human preference rankings, then uses PPO to optimize the LLM to score highly on the reward model. This aligns the model to be helpful, harmless, and honest."},
    {"id": "doc_13", "text": "DPO (Direct Preference Optimization) is an alternative to RLHF that skips the reward model entirely. It directly optimizes the LLM on preference pairs using a simple classification loss. More stable training and simpler to implement than RLHF with PPO."},
    {"id": "doc_14", "text": "Vector databases store high-dimensional embedding vectors and support fast approximate nearest neighbor search. Examples include Chroma, Pinecone, Weaviate, and pgvector. They are the storage backbone of production RAG systems."},
]

# ─────────────────────────────────────────
# STEP 3: build ChromaDB index
# ─────────────────────────────────────────
print("building ChromaDB index...")
chroma  = chromadb.Client()

# delete collection if exists (clean start)
try:
    chroma.delete_collection("ai_knowledge")
except:
    pass
collection = chroma.create_collection(
    "ai_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# embed all documents
texts = [d["text"] for d in DOCUMENTS]
ids   = [d["id"]   for d in DOCUMENTS]

t0          = time.time()
embeddings  = embedder.encode(texts, show_progress_bar=False).tolist()
embed_time  = time.time() - t0

collection.add(documents=texts, embeddings=embeddings, ids=ids)
print(f"✓ {len(DOCUMENTS)} documents indexed in {embed_time:.2f}s")
print(f"  embedding dim: {len(embeddings[0])}\n")

# ─────────────────────────────────────────
# STEP 4: RAG query function
# ─────────────────────────────────────────
def rag_query(question, top_k=3, verbose=True):
    # embed the question
    q_embedding = embedder.encode([question]).tolist()

    # retrieve from ChromaDB
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=top_k
    )

    docs    = results["documents"][0]
    dists   = results["distances"][0]

    if verbose:
        print(f"── retrieved for: '{question}' ──")
        for dist, doc in zip(dists, docs):
            similarity = 1 - dist   # chroma returns distance, convert to similarity
            print(f"  [sim {similarity:.3f}] {doc[:75]}...")

    # build context
    context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)])

    prompt = f"""Answer the question using ONLY the numbered context passages below.
Do not use any knowledge outside the provided context.
If the answer is not found in the context, respond with exactly:
'Not found in knowledge base.'

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        max_tokens=300,
        messages=[
            {"role": "system",
             "content": "Answer strictly from the provided context passages only."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ─────────────────────────────────────────
# STEP 5: test queries
# ─────────────────────────────────────────
questions = [
    "How does self-attention work mathematically?",
    "What is LoRA and why is it useful for fine-tuning?",
    "Why does validation loss increase while training loss decreases?",
    "What is the difference between LayerNorm and BatchNorm?",
    "What is QLoRA and how does it differ from LoRA?",       # new — tests doc_10
    "How does RLHF differ from DPO?",                        # new — tests doc_12/13
    "What is the capital of Australia?",                     # out of scope
]

for q in questions:
    print(f"\n{'═'*55}")
    print(f"Q: {q}")
    print(f"{'─'*55}")
    answer = rag_query(q, verbose=True)
    print(f"\nA: {answer}")

# ─────────────────────────────────────────
# STEP 6: show embedding similarity directly
# ─────────────────────────────────────────
print(f"\n{'═'*55}")
print("EMBEDDING SIMILARITY DEMO")
print("═"*55)
pairs = [
    ("self-attention mechanism", "Query Key Value matrices attention"),
    ("self-attention mechanism", "capital of Australia"),
    ("overfitting training validation", "model memorizes training data"),
]
for a, b in pairs:
    ea = embedder.encode([a])[0]
    eb = embedder.encode([b])[0]
    sim = sum(x*y for x,y in zip(ea,eb)) / (
          sum(x*x for x in ea)**0.5 * sum(x*x for x in eb)**0.5)
    print(f"\n'{a[:40]}'\n'{b[:40]}'\n  → similarity: {sim:.4f}")