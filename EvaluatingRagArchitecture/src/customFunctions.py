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

    embedding_model = SentenceTransformer('models/embeddings_model/all_MiniLM_L6_v2/')
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
    embedding_model = SentenceTransformer('models/embeddings_model/all_MiniLM_L6_v2/')
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

    if hasattr(query_results, 'points'):
        context = "\n".join([f"- {r.payload[colname]}" for r in query_results.points])
    else:
        context = "\n".join([f"- {r.payload[colname]}" for r in query_results])

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
    if resp.status_code == 200:
        answer = resp.json()['text']
        if verbose:
            print(f"\nGenerated answer: {answer}")
    else:
        if verbose:
            print(f"LLM error: {resp.status_code}")
    if len(answer)>10:
        answer = answer[len(prompt):]
    return answer

def evaluate_answer(generated_answer, ground_truth, query_results, groundContext="",colname='doc'):
    import evaluate
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    
    embedding_model = SentenceTransformer('models/embeddings_model/all_MiniLM_L6_v2/')
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

    if hasattr(query_results, 'points'):
        retrieval_scores = [r.score for r in query_results.points]
    else:
        retrieval_scores = [r.score for r in query_results]
    avg_retrieval_score = np.mean(retrieval_scores) if retrieval_scores else 0.0
    
    avg_retrieval_sim = 0.0
    if len(groundContext)>0:
        if hasattr(query_results, 'points'):
            retrieval_similarity = [cosine_similarity([embedding_model.encode(rc)], embedding_model.encode(groundContext).reshape(1, -1))[0][0] for r in query_results.points for rc in r.payload[colname]]
        else:
            retrieval_similarity = [cosine_similarity([embedding_model.encode(rc)], embedding_model.encode(groundContext).reshape(1, -1))[0][0] for r in query_results for rc in r.payload[colname]]
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

def visualizeMetrics(metricList, paramValues, paramName):
    import matplotlib.pyplot as plt
    import numpy as np

    mean_metrics = []
    for result in metricList:
        mean_metrics.append(result.mean().to_dict())

    max_metrics = []
    for result in metricList:
        max_metrics.append(result.max().to_dict())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Group 1: ROUGE metrics
    rouge_metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    for metric in rouge_metrics:
        values = [m[metric] for m in mean_metrics]
        axes[0].plot(paramValues, values, marker='o', label=metric)
    axes[0].set_title('ROUGE Metrics')
    axes[0].set_xlabel(paramName)
    axes[0].set_ylabel('Score')
    axes[0].legend()
    axes[0].grid(True)

    # Group 2: Semantic similarity metrics
    semantic_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1', 'cosine_similarity']
    for metric in semantic_metrics:
        values = [m[metric] for m in mean_metrics]
        axes[1].plot(paramValues, values, marker='o', label=metric)
    axes[1].set_title('Semantic Similarity Metrics')
    axes[1].set_xlabel(paramName)
    axes[1].set_ylabel('Similarity')
    axes[1].legend()
    axes[1].grid(True)

    # Group 3: Generation quality metrics
    generation_metrics = ['bleu', 'rougeL']
    for metric in generation_metrics:
        values = [m[metric] for m in mean_metrics]
        axes[2].plot(paramValues, values, marker='o', label=metric)
    axes[2].set_title('Text Generation Quality')
    axes[2].set_xlabel(paramName)
    axes[2].set_ylabel('Score')
    axes[2].legend()
    axes[2].grid(True)

    # Group 4: Retrieval quality metrics
    retrieval_metrics = ['avg_retrieval_score', 'avg_retrieval_similarity']
    for metric in retrieval_metrics:
        values = [m[metric] for m in mean_metrics]
        axes[3].plot(paramValues, values, marker='o', label=metric)
        values = [m[metric] for m in max_metrics]
        axes[3].plot(paramValues, values, marker='o', label=metric.replace('avg', 'max'))
    axes[3].set_title('Retrieval Quality')
    axes[3].set_xlabel(paramName)
    axes[3].set_ylabel('Score')
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()

def plot_single_metric(ax, metric_name, mean_data, std_data=None, title=None, ylabel=None):
    import matplotlib.pyplot as plt
    
    if metric_name == "rouge1":
        title = "Rouge1 Score"
    elif metric_name == "rouge2":
        title = "Rouge2 Score"
    elif metric_name == "rougeL":
        title = "RougeL Score"
    elif metric_name == "rougeLsum":
        title = "RougeLsum Score"
    elif metric_name == "bertscore_precision":
        title = "BERTScore Precision"
    elif metric_name == "bertscore_recall":
        title = "BERTScore Recall"
    elif metric_name == "bertscore_f1":
        title = "BERTScore F1"
    elif metric_name == "bleu":
        title = "BLEU Score"
    elif metric_name == "cosine_similarity":
        title = "Cosine Similarity"
    elif metric_name == "avg_retrieval_score":
        title = "Avg Retrieval Score"
    elif metric_name == "avg_retrieval_similarity":
        title = "Avg Retrieval Similarity"
    else:
        title = metric_name

    if ylabel is None:
        ylabel = metric_name
    num_archs = len(mean_data)
    colors = plt.cm.Set3(range(num_archs))
    if std_data is not None:
        bars=ax.bar(range(num_archs),mean_data.values,color=colors,yerr=std_data.values,capsize=5,error_kw={'elinewidth':1,'ecolor':'black','alpha':0.7})
    else:
        bars = ax.bar(range(num_archs), mean_data.values, color=colors)
    ax.set_xticks(range(num_archs))
    ax.set_xticklabels(mean_data.index, rotation=45, ha='right', fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mean_data.values):
        y_pos = v
        if std_data is not None:
            y_pos = v + std_data.iloc[i]
        ax.text(i, y_pos, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    if std_data is not None:
        max_value = (mean_data.values + std_data.values).max()
        min_value = max(0, (mean_data.values - std_data.values).min())
        if metric_name.lower().find("bert") == -1:
            ax.set_ylim(min_value * 0.95 if min_value > 0 else 0, max_value * 1.05)
        else:
            ax.set_ylim(min_value * 0.995 if min_value > 0 else 0, max_value * 1.005)
    else:
        ax.set_ylim(mean_data.values.min() * 0.95, mean_data.values.max() * 1.05)

def compareViz(arch_names, arch_dfs, error_type='std'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    combined_data = pd.DataFrame()
    for i, (name, df) in enumerate(zip(arch_names, arch_dfs)):
        df_copy = df.copy()
        df_copy['Architecture'] = name
        df_copy['Arch_ID'] = i
        combined_data = pd.concat([combined_data, df_copy], ignore_index=True)
    rouge_metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    bertscore_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    other_metrics = ['bleu', 'cosine_similarity', 'avg_retrieval_score', 'avg_retrieval_similarity']
    all_metrics = rouge_metrics + bertscore_metrics + other_metrics
    mean_scores = combined_data.groupby('Architecture', sort=False)[all_metrics].mean()
    if error_type == 'std':
        error_scores = combined_data.groupby('Architecture', sort=False)[all_metrics].std()
    elif error_type == 'sem':  # Standard Error of the Mean
        error_scores = combined_data.groupby('Architecture', sort=False)[all_metrics].sem()
    elif error_type == 'ci':  # 95% Confidence Interval
        error_scores = combined_data.groupby('Architecture', sort=False)[all_metrics].std() / np.sqrt(combined_data.groupby('Architecture', sort=False)[all_metrics].count()) * 1.96
    else:
        error_scores = None
    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
    metrics_to_plot = all_metrics[:12]
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        if error_scores is not None:
            plot_single_metric(ax, metric, mean_scores[metric], error_scores[metric])
        else:
            plot_single_metric(ax, metric, mean_scores[metric])
    plt.suptitle('RAG Architecture Comparison Across All Metrics', fontsize=16, fontweight='bold', y=0.92)
    plt.tight_layout()
    plt.show()
    return mean_scores, error_scores

def corrMatrix(architectures):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    all_results = pd.DataFrame()
    for arch_name, df in architectures.items():
        df_copy = df.copy()
        df_copy['Architecture'] = arch_name
        all_results = pd.concat([all_results, df_copy], ignore_index=True)
    metric_columns = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum','bertscore_precision', 'bertscore_recall', 'bertscore_f1',
        'bleu', 'cosine_similarity','avg_retrieval_score', 'avg_retrieval_similarity']
    correlation_matrix = all_results[metric_columns].corr(method='pearson')
    p_values = pd.DataFrame(index=metric_columns, columns=metric_columns)
    for i in metric_columns:
        for j in metric_columns:
            if i != j:
                corr, p_val = pearsonr(all_results[i].dropna(), all_results[j].dropna())
                p_values.loc[i, j] = p_val
            else:
                p_values.loc[i, j] = 1.0
    plt.figure(figsize=(6, 5))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(correlation_matrix,mask=mask,cmap=cmap,vmax=1,vmin=-1, center=0, annot=True,fmt='.2f',square=True, linewidths=0.1,cbar_kws={"shrink": 0.2},annot_kws={"size": 7})
    pretty_names = {
        'rouge1': 'ROUGE-1',
        'rouge2': 'ROUGE-2',
        'rougeL': 'ROUGE-L',
        'rougeLsum': 'ROUGE-LSum',
        'bertscore_precision': 'BERTScore P',
        'bertscore_recall': 'BERTScore R',
        'bertscore_f1': 'BERTScore F1',
        'bleu': 'BLEU',
        'cosine_similarity': 'Cosine Sim',
        'avg_retrieval_score': 'Retrieval Score',
        'avg_retrieval_similarity': 'Retrieval Similarity'
    }
    tick_labels = [pretty_names.get(col, col) for col in correlation_matrix.columns]
    plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=45, ha='right', fontsize=6)
    plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=0, fontsize=6)
    plt.title('Correlation Matrix of Evaluation Metrics\nAcross All RAG Architectures (n=150 samples)', 
            fontsize=8, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.show()