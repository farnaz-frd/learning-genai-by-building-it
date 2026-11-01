# rag.py — RAG pipeline from scratch, no frameworks
from openai import OpenAI
import json, re, math

client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
MODEL  = "llama3.1:8b"

# ─────────────────────────────────────────
# STEP 1: your knowledge base
# a small corpus about AI concepts you know well
# ─────────────────────────────────────────
DOCUMENTS = [
    {"id": 0, "text": "The Transformer architecture was introduced in 2017 in the paper 'Attention is All You Need' by Vaswani et al. It replaced recurrent networks with self-attention mechanisms, enabling parallel processing of sequences."},
    {"id": 1, "text": "Self-attention computes Query, Key, and Value matrices from the input. The attention score between tokens i and j is computed as Q_i dot K_j divided by sqrt(d_k), then softmax is applied to get weights for the Value vectors."},
    {"id": 2, "text": "LoRA (Low-Rank Adaptation) is a fine-tuning technique that freezes the original model weights and adds trainable low-rank matrices A and B. The weight update is expressed as delta_W = B times A, where the rank r is much smaller than the original dimensions."},
    {"id": 3, "text": "Large Language Models are trained on next-token prediction. Given a sequence of tokens, the model learns to predict the most probable next token. This simple objective forces the model to learn grammar, facts, reasoning and world knowledge."},
    {"id": 4, "text": "Overfitting occurs when training loss decreases but validation loss increases. The model memorizes training data instead of learning general patterns. Solutions include dropout, regularization, early stopping, and getting more training data."},
    {"id": 5, "text": "RAG (Retrieval-Augmented Generation) combines a retrieval system with a language model. Relevant documents are retrieved from a knowledge base and injected into the LLM prompt as context, reducing hallucination and keeping knowledge up to date."},
    {"id": 6, "text": "Multi-head attention runs h independent attention operations in parallel. Each head projects the input into a lower-dimensional subspace using separate Wq, Wk, Wv matrices. Heads learn to specialize in different relationship types through training."},
    {"id": 7, "text": "The causal mask in decoder-only transformers prevents each token from attending to future tokens. It is implemented as an upper-triangular matrix of negative infinity values added to attention scores before softmax."},
    {"id": 8, "text": "Positional encoding adds position information to token embeddings since attention has no inherent sense of order. Sinusoidal encodings use sine and cosine functions at different frequencies. RoPE encodes relative position by rotating query and key vectors."},
    {"id": 9, "text": "LayerNorm normalizes activations over the last dimension (per token, per position). Unlike BatchNorm it is independent of batch size and sequence length, making it suitable for variable-length sequences and batch size of 1 at inference."},
]

# ─────────────────────────────────────────
# STEP 2: embedding — TF-IDF style (no model needed)
# in a real system you'd use sentence-transformers here
# ─────────────────────────────────────────
def tokenize(text):
    return re.findall(r'\w+', text.lower())

def build_tfidf(docs):
    N = len(docs)
    df = {}  # document frequency
    tf_docs = []
    for doc in docs:
        tokens = tokenize(doc["text"])
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        tf_docs.append(tf)
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    # compute TF-IDF vectors
    vocab  = list(df.keys())
    vecs   = []
    for tf in tf_docs:
        vec = []
        for term in vocab:
            tf_val  = tf.get(term, 0) / max(sum(tf.values()), 1)
            idf_val = math.log(N / (df.get(term, 1)))
            vec.append(tf_val * idf_val)
        vecs.append(vec)
    return vocab, vecs

def cosine_similarity(a, b):
    dot   = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    return dot / (norm_a * norm_b + 1e-9)

def retrieve(query, vocab, vecs, docs, top_k=3):
    # embed the query with same TF-IDF vocab
    qt = tokenize(query)
    N  = len(docs)
    df = {v: sum(1 for vec_tf in
          [tokenize(d["text"]) for d in docs] if v in vec_tf) for v in vocab}
    q_vec = []
    for term in vocab:
        tf_val  = qt.count(term) / max(len(qt), 1)
        idf_val = math.log(N / (df.get(term, 1) + 1e-9))
        q_vec.append(tf_val * idf_val)

    scores = [(cosine_similarity(q_vec, vec), docs[i])
              for i, vec in enumerate(vecs)]
    scores.sort(key=lambda x: -x[0])
    return scores[:top_k]

# ─────────────────────────────────────────
# STEP 3: build the index
# ─────────────────────────────────────────
print("building retrieval index...")
vocab, vecs = build_tfidf(DOCUMENTS)
print(f"vocab size: {len(vocab)} terms, {len(DOCUMENTS)} documents indexed\n")

# ─────────────────────────────────────────
# STEP 4: RAG query function
# ─────────────────────────────────────────
def rag_query(question, top_k=3, verbose=True):
    # retrieve
    results = retrieve(question, vocab, vecs, DOCUMENTS, top_k)

    if verbose:
        print(f"── retrieved docs for: '{question}' ──")
        for score, doc in results:
            print(f"  [score {score:.3f}] {doc['text'][:80]}...")

    # build context
    context = "\n\n".join([f"[{i+1}] {doc['text']}"
                           for i, (_, doc) in enumerate(results)])

    # generate
    prompt = f"""Answer using ONLY the numbered context passages. 
    Do not use any knowledge outside the provided context.
    If the context does not contain the answer, respond with exactly: 
    'Not found in knowledge base.'"

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        max_tokens=256,
        messages=[
            {"role": "system", "content": "You are a precise technical assistant. Answer only from the provided context."},
            {"role": "user",   "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ─────────────────────────────────────────
# STEP 5: test with real questions
# ─────────────────────────────────────────
questions = [
    "How does self-attention work mathematically?",
    "What is LoRA and why is it useful for fine-tuning?",
    "Why does validation loss increase while training loss decreases?",
    "What is the difference between LayerNorm and BatchNorm?",
    "What is the capital of Australia?",   # not in knowledge base → should say so
]

for q in questions:
    print(f"\n{'═'*55}")
    print(f"Q: {q}")
    print(f"{'─'*55}")
    answer = rag_query(q, verbose=True)
    print(f"\nA: {answer}")