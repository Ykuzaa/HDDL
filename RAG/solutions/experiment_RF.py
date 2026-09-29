# Once again for the prompt below compare the answer with and without RAG
        # prompt_RF='Is Robert Redford alive ?; Answer must be [Yes] or [No]'

# Print what is the retrieved context

query = prompt_RF

response_no_RAG = rag.foundation_model.generate_response(prompt=query)

context = rag.get_retrieval(query=query,number_of_hits=3)

print("Retrieved context:", context, "\n")

response_RAG = rag.generate_response_rag(query=query)

print(response_no_RAG,"\n",response_RAG)
