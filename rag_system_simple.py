import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import glob

print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# Create Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

# Delete existing collection if it exists
collection_name = "knowledge_base"
try:
    client.delete_collection(collection_name)
except:
    pass

collection = client.create_collection(
    name=collection_name,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
)

print("Vector database created!")

def chunk_text(text, chunk_size=500):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 100):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Create knowledge base folder
os.makedirs("./knowledge_base", exist_ok=True)

# Create sample documents if they don't exist
if not glob.glob("./knowledge_base/*.txt"):
    # Create security policy document
    with open("./knowledge_base/security_policy.txt", 'w') as f:
        f.write("Password Policy\n")
        f.write("- Passwords must be at least 12 characters long\n")
        f.write("- Passwords must contain uppercase, lowercase, numbers, and special characters\n")
        f.write("- Users must change passwords every 90 days\n")
        f.write("- Password reuse is not allowed for the last 5 passwords\n")
        f.write("- Failed login attempts lock account for 15 minutes after 5 attempts\n")
    
    with open("./knowledge_base/api_guide.txt", 'w') as f:
        f.write("API Authentication\n")
        f.write("- Use Bearer tokens for authentication\n")
        f.write("- Tokens expire after 30 minutes\n")
        f.write("- To get a token, POST username/password to /token endpoint\n")
        f.write("- Include token in Authorization header: Bearer <token>\n")

    with open("./knowledge_base/faq.txt", 'w') as f:
        f.write("Frequently Asked Questions\n")
        f.write("Q: How do I reset my password?\n")
        f.write("A: Go to Settings -> Security -> Reset Password\n")
        f.write("Q: My token expired, what do I do?\n")
        f.write("A: Get a new token by logging in again at /token endpoint\n")
        f.write("Q: How many login attempts before lockout?\n")
        f.write("A: 5 failed attempts will lock your account for 15 minutes\n")

# Load knowledge base
def load_knowledge_base(folder_path):
    documents = []
    metadatas = []
    ids = []
    
    file_paths = glob.glob(f"{folder_path}/*.txt")
    
    for file_idx, file_path in enumerate(file_paths):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        source = os.path.basename(file_path).replace('.txt', '')
        chunks = chunk_text(content)
        
        for chunk_idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": source, "chunk": chunk_idx})
            ids.append(f"{source}_{chunk_idx}")
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Loaded {len(documents)} chunks from {len(file_paths)} files")

load_knowledge_base("./knowledge_base")

def search(query, top_k=3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results['documents'][0], results['metadatas'][0], results['distances'][0]

def answer_question(query):
    docs, metadatas, distances = search(query, top_k=3)
    print(f"\nQuestion: {query}")
    print(f"\nRetrieved from: {metadatas[0]['source']}")
    print(f"\nAnswer:")
    print("-" * 50)
    print(docs[0])
    print("-" * 50)
    print(f"\nConfidence: {(1 - distances[0]) * 100:.1f}%")
    return docs[0]

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RAG SYSTEM - CUSTOMER SUPPORT")
    print("="*60)
    
    test_questions = [
        "How long must passwords be?",
        "How do I get an API token?",
        "What happens after 5 failed login attempts?",
        "How do I reset my password?"
    ]
    
    for question in test_questions:
        answer_question(question)
        print("\n" + "-"*60)
