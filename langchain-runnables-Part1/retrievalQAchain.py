from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

#Load the document
loader = TextLoader("docs.txt") #Ensure docs.txt exists
documents = loader.load()

#Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert text into embeddings and store in FAISS
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# Create a retriever (fetches relavant documents)
retriever = vectorstore.as_retriever()

#Manually Retrieve Relevant Documents
query = "What are the key takeaway from the document?"
retrieved_docs = retriever.get_relevant_documents(query)

#Combine Retrieved Text into a Single Prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Initialize the LLM
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

#Create RetrievalQAChain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Manually Pass Retrieved Text to LLM 
query= "What are the key takeaways from the documents?"
answer = qa_chain.run(query)

#Print the Answer
print("Answer:", answer)