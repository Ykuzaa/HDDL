# The cross-encoder (a small model trained to score (query, passage) pairs)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=str(device))

# 1. We retrieve more hits than needed with our retriever (bi-encoder)
retrieved_info = rag.retriever.search_best(query=query, number_of_hits=5)
candidates = [chunk.content for (Id, chunk, sim) in retrieved_info]
cos_sims = [sim for (Id, chunk, sim) in retrieved_info]

# 2. We rerank them with the cross-encoder
scores = reranker.predict([(query, c) for c in candidates])

reranked = sorted(zip(candidates, cos_sims, scores), key=lambda x: x[2], reverse=True)
df_rerank = pd.DataFrame(reranked, columns=["chunk", "cos sim (bi-encoder)", "cross-encoder score"])
print(df_rerank)

# 3. We keep the best chunks and pass them to the decoder together with the query
top_chunks = [c for (c, _, _) in reranked[:2]]
response_rerank = rag.foundation_model.generate_response_with_context(prompt=query, context=top_chunks)
print(short_response(response_rerank))
