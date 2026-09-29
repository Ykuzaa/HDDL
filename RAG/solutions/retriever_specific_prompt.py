# Unit test

# Recall that we have already in our previous unit tests defined an embedding model and a splitter

        # embed_model=EmbeddingModel(EMBEDD_MODEL_PATH=Embed_mini)

        # Split=Splitter(embed_model)
        # Split.chunks=[]
        # Split.get_chunks(path_doc=this_path,chunk_size=30,overlap=15,sentence_split=True)

print(Split.docs)

chunks=Split.chunks

print(len(chunks))

retriever = Retriever(embed_model)


# Add the chunks to the index

# Get the best results using your retriever to the query 

        #prompt_RF='Is Robert Redford alive ?; Answer must be [Yes] or [No]'


retriever.add_elements_to_index(chunks=chunks)

query='Is Robert Redford alive'

results=retriever.search_best(query=query,number_of_hits=3,adapt=True)

print("results",results)
