from langchain_community.document_loaders import YoutubeLoader
from langchain_core.messages import HumanMessage, AIMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
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
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.\n"
            "Use the provided context to answer factual questions about the video.\n"
            "You may also use the conversation history to correct, clarify, or acknowledge mistakes in your own previous responses.\n"
            "Do NOT invent facts that are not present in the context or chat history.\n"
            "If the answer cannot be found in either the context or the conversation history, say you don't know and explain why."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        ),
    ]
)

# 3. Formatting helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chat_history = []

# 4. The Chain (LCEL)
rag_chain = (
    RunnableParallel(
        {
            "context": RunnableLambda(lambda x: x["question"])
            | retriever
            | RunnableLambda(format_docs),
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)


print("Enter Your Questions about the YouTube Video (type 'exit' to quit):")

while True:
    query = input("You: ")

    if query.lower() in ["exit", "quit", "q"]:
        print("Exiting...")
        break

    response = rag_chain.invoke(
        {
            "question": query,
            "chat_history": chat_history,
        }
    )

    # Update chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response))

    print("-" * 50)
    print(f"QUESTION: {query}")
    print("-" * 50)
    print(f"ANSWER:\n{response}")
    print("-" * 50)
