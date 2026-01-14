def loadfileDB(collection_name, filepath, colname='doc', chunk_size=1000):
    import json
    import pandas as pd
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer

    print("\nSetting up Qdrant...")
    client = QdrantClient(host="localhost", port=6333, timeout=10)
    if not client.collection_exists(collection_name):
        print("creating collection...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    points = client.get_collection(collection_name).points_count
    if points > 10:
        print("Collection already exists and has data. Skipping upload.", points, "points found.")
        return client

    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    global_id = 0
    
    for chunk_idx, df_chunk in enumerate(pd.read_json(filepath, lines=True, chunksize=chunk_size)):
        if ((chunk_idx + 1) % 10 == 0):
            print(f"Processing chunk {chunk_idx + 1}...")
        texts = []
        contexts_list = []
        for idx, row in df_chunk.iterrows():
            contexts = row[colname]
            contexts_list.append(contexts)
            if isinstance(contexts, list):
                context_text = " ".join(str(ctx) for ctx in contexts)
            else:
                context_text = str(contexts)
            texts.append(context_text)
        if texts:
            chunk_embeddings = embedding_model.encode(texts, show_progress_bar=False)
            chunk_points = []
            for j in range(len(chunk_embeddings)):
                chunk_points.append(PointStruct(
                    id=global_id,
                    vector=chunk_embeddings[j].tolist(),
                    payload={colname: contexts_list[j] if isinstance(contexts_list[j], list) else [contexts_list[j]]}
                ))
                global_id += 1
            client.upsert(collection_name=collection_name, points=chunk_points)    
    print(f"\n✅ Completed! Total uploaded {global_id} documents")
    return client

def retrieveQueryEmbeddings(query_texts,client,collection_name, colname='doc', k=3,verbose=True):
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode(query_texts).tolist()
    results = client.query_points(collection_name=collection_name,query=query_embedding,limit=k,with_payload=True)
    if verbose:
        for i, point in enumerate(results.points):
            print(f"\n{i+1}. Score: {point.score:.3f}")
            print(f"   ID: {point.id}")
            print(f"   {colname}: {point.payload.get(colname, 'No text')[:150]}...")
    return results

def llmRespose(query_results, test_query, colname='doc', verbose=True):
    import httpx

    context = "\n".join([f"- {r.payload[colname]}" for r in query_results.points])
    prompt = f"""Based on the following context, answer the question.

    Context:
    {context}

    Question: {test_query}

    Answer:"""

    resp = httpx.post(
        "http://localhost:8001/generate",
        json={"prompt": prompt, "max_tokens": 150, "temperature": 0.7},
        timeout=120.0
    )
    answer=""
    if verbose:
        if resp.status_code == 200:
            answer = resp.json()['text']
            print(f"\nGenerated answer: {answer}")
        else:
            print(f"LLM error: {resp.status_code}")
    if len(answer)>10:
        answer = answer[len(prompt):]
    return answer

def evaluate_answer(generated_answer, ground_truth, query_results, groundContext="",colname='doc'):
    import evaluate
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    bleu = evaluate.load("bleu")

    if isinstance(ground_truth, list):
        ground_truth = " ".join(str(ctx) for ctx in ground_truth)
    else:
        ground_truth = str(ground_truth)

    rouge_scores = rouge.compute(predictions=[generated_answer],references=[ground_truth])
    bert_scores = bertscore.compute(predictions=[generated_answer],references=[ground_truth],lang="en")
    bleu_scores = bleu.compute(predictions=[generated_answer],references=[ground_truth])
    similarity = cosine_similarity([embedding_model.encode(generated_answer)], [embedding_model.encode(ground_truth)])[0][0]

    retrieval_scores = [r.score for r in query_results.points]
    avg_retrieval_score = np.mean(retrieval_scores) if retrieval_scores else 0.0
    
    avg_retrieval_sim = 0.0
    if len(groundContext)>0:
        retrieval_similarity = [cosine_similarity([embedding_model.encode(rc)], embedding_model.encode(groundContext).reshape(1, -1))[0][0] for r in results.points for rc in r.payload[colname]]
        avg_retrieval_sim = np.mean(retrieval_similarity) 

    metrics = {
        "rouge1": rouge_scores['rouge1'],
        "rouge2": rouge_scores['rouge2'],
        "rougeL": rouge_scores['rougeL'],
        "rougeLsum": rouge_scores['rougeLsum'],
        "bertscore_precision": bert_scores['precision'][0],
        "bertscore_recall": bert_scores['recall'][0],
        "bertscore_f1": bert_scores['f1'][0],
        "bleu": bleu_scores['bleu'],
        "cosine_similarity": float(similarity),
        "avg_retrieval_score": avg_retrieval_score,
        "avg_retrieval_similarity": avg_retrieval_sim
    }
    return metrics
