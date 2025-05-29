import os
import time
import numpy as np
import streamlit as st
from pathlib import Path
from faiss import IndexFlatL2
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from pypdf import PdfReader


@st.cache_resource
def get_client():
    """Returns a cached instance of the Mistral client."""
    api_key = os.environ["MISTRAL_API_KEY"]
    return Mistral(api_key=api_key)

CLIENT = get_client()

system_prompt = """
Tu es GriotBot, un assistant inspiré des sages et griots du Bénin.

---------------------
{context}
---------------------

À partir de ce contexte culturel (proverbes, contes, maximes ou conseils traditionnels africains), réponds à la requête suivante.

Ta réponse doit :
- Commencer par un proverbe, conte ou maxime tiré du contexte.
- Suivre d'une interprétation ou d'un conseil pertinent selon la situation exprimée.
- Utiliser un ton chaleureux, imagé et empreint de sagesse.
- Ne parler que le français.
- Respecter fidèlement les traditions et valeurs africaines.
- Ne jamais inventer de contenu absent du contexte, sauf si explicitement demandé.

Si le contexte ne fournit pas assez d'information, décline poliment de répondre.

Requête de l'utilisateur : {query}

Réponse :
"""


# Initialize session state variables if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to add a message to the chat
def add_message(msg, agent="ai", stream=True, store=True):
    """Adds a message to the chat interface, optionally streaming the output."""
    if stream and isinstance(msg, str):
        msg = stream_str(msg)

    with st.chat_message(agent):
        if stream:
            output = st.write_stream(msg)
        else:
            output = msg
            st.write(msg)

    if store:
        st.session_state.messages.append(dict(agent=agent, content=output))

# Function to stream a string with a delay
def stream_str(s, speed=250):
    """Yields characters from a string with a delay to simulate streaming."""
    for c in s:
        yield c
        time.sleep(1 / speed)


# Function to stream the response from the AI
def stream_response(response):
    """Yields responses from the AI, replacing placeholders as needed."""
    for chunk in response:
        if chunk.data.choices[0].delta.content:
            content = chunk.data.choices[0].delta.content
            # prevent $ from rendering as LaTeX
            content = content.replace("$", "\$")
            yield content


# Decorator to cache the embedding computation with rate limiting
@st.cache_data
def embed(text: str, max_retries=3, base_delay=1):
    """Returns the embedding for a given text, caching the result with rate limiting."""
    for attempt in range(max_retries):
        try:
            response = CLIENT.embeddings.create(
                model="mistral-embed",
                inputs=[text]
            )
            return response.data[0].embedding
        except SDKError as e:
            if "429" in str(e):  # Rate limit error
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                st.warning(f"Rate limit reached. Waiting {delay} seconds...")
                time.sleep(delay)
                if attempt == max_retries - 1:
                    st.error("Rate limit persistant. Veuillez réessayer plus tard.")
                    raise
            else:
                raise
        except Exception as e:
            st.error(f"Erreur lors de la génération d'embedding: {e}")
            raise


# Function to create embeddings in batches to avoid rate limits
def create_embeddings_batch(chunks, batch_size=5, delay_between_batches=2):
    """Create embeddings in batches to respect rate limits."""
    embeddings = []
    total_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        status_text.text(f"Traitement du lot {batch_num}/{total_batches}...")
        
        batch_embeddings = []
        for chunk in batch:
            embedding = embed(chunk)
            batch_embeddings.append(embedding)
            time.sleep(0.1)  # Small delay between individual requests
        
        embeddings.extend(batch_embeddings)
        progress_bar.progress(min(1.0, (i + batch_size) / len(chunks)))
        
        # Delay between batches except for the last one
        if i + batch_size < len(chunks):
            time.sleep(delay_between_batches)
    
    progress_bar.empty()
    status_text.empty()
    return embeddings


# Function to build and cache the index from PDFs in a directory
@st.cache_resource
def build_and_cache_index():
    """Builds and caches the index from PDF documents in the specified directory."""
    pdf_files = list(Path("data").glob("*.pdf"))
    
    if not pdf_files:
        st.error("Aucun fichier PDF trouvé dans le dossier 'data'")
        return None, None
    
    st.info(f"Traitement de {len(pdf_files)} fichier(s) PDF...")
    text = ""

    for pdf_file in pdf_files:
        st.info(f"Lecture de {pdf_file.name}...")
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + "\n\n"

    if not text.strip():
        st.error("Aucun texte extractible trouvé dans les PDFs")
        return None, None

    chunk_size = 500
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    st.info(f"Création des embeddings pour {len(chunks)} segments de texte...")

    # Create embeddings in batches to respect rate limits
    embeddings = create_embeddings_batch(chunks)
    
    embeddings = np.array(embeddings)
    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(embeddings)

    st.success("Index créé avec succès!")
    return index, chunks


# Function to reply to queries using the built index
def reply(query: str, index: IndexFlatL2, chunks):
    """Generates a reply to the user's query based on the indexed PDF content."""
    try:
        embedding = embed(query)
        embedding = np.array([embedding])

        _, indexes = index.search(embedding, k=2)
        context = [chunks[i] for i in indexes.tolist()[0]]

        # Utilisation de la nouvelle API Mistral
        messages = [
            {
                "role": "user", 
                "content": system_prompt.format(context=context, query=query)
            }
        ]
        
        response = CLIENT.chat.stream(
            model="mistral-medium",
            messages=messages
        )
        
        add_message(stream_response(response))
        
    except SDKError as e:
        if "429" in str(e):
            add_message("⏳ Limite de taux atteinte. Veuillez patienter quelques instants avant de réessayer.", stream=False)
        else:
            add_message(f"❌ Erreur API: {e}", stream=False)
    except Exception as e:
        add_message(f"❌ Erreur inattendue: {e}", stream=False)


# Main application logic
def main():
    """Main function to run the application logic."""
    st.title("🎭 GriotBot - Assistant des Sages du Bénin")
    
    if st.sidebar.button("🔴 Reset conversation"):
        st.session_state.messages = []

    # Check if data directory exists
    if not Path("data").exists():
        st.error("📁 Le dossier 'data' n'existe pas. Veuillez le créer et y placer vos fichiers PDF.")
        return

    try:
        index, chunks = build_and_cache_index()
        
        if index is None or chunks is None:
            st.error("❌ Impossible de créer l'index. Vérifiez vos fichiers PDF.")
            return

        for message in st.session_state.messages:
            with st.chat_message(message["agent"]):
                st.write(message["content"])

        query = st.chat_input("Posez une question sur vos documents PDF...")

        if not st.session_state.messages:
            add_message("🙏 Bienvenue ! Je suis GriotBot, inspiré des sages du Bénin. Posez-moi vos questions !")

        if query:
            add_message(query, agent="human", stream=False, store=True)
            reply(query, index, chunks)
            
    except Exception as e:
        st.error(f"❌ Erreur dans l'application: {e}")


if __name__ == "__main__":
    main()