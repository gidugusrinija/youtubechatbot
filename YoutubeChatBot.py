from langchain_community.document_loaders import YoutubeLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# The YouTube Video URL you want to chat with
VIDEO_URL = "https://www.youtube.com/watch?v=1E6mBvF_hKg"

loader = YoutubeLoader.from_youtube_url(
    VIDEO_URL,
    add_video_info=False,
    language=["en", "en-US"]
)
docs = loader.load()

print("...Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} chunks.")

print("Generating embeddings and storing in ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    collection_name="youtube_rag"
)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "lambda_mult": 0.5})
print("...Vector Store Ready...")

print("...Building RAG Chain...")

# 1. The LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. The Prompt (Augmentation: Context + Question)
template = """Answer the question based only on the following context:
{context}
Question: {question}
if the question can't be answered based on the context, say "I don't know and explain the reason"
"""
prompt = ChatPromptTemplate.from_template(template)


# 3. Formatting helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 4. The Chain (LCEL)
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


print("...Asking AI...")
while True:
    query = input()
    if query.lower() in ["exit", "quit", "q"]:
        print("Exiting...")
        break
    response = rag_chain.invoke(query)

    print("-" * 50)
    print(f"QUESTION: {query}")
    print("-" * 50)
    print(f"ANSWER:\n{response}")
    print("-" * 50)
