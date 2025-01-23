import validators
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from yt_dlp import YoutubeDL
from langchain.schema import Document  

### Streamlit App Configuration
st.set_page_config(
    page_title="LangChain Summarizer",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded",
)

### Sidebar Styling and Content
with st.sidebar:
    st.title("LangChain Summarizer")
    st.image(
        "https://imgs.search.brave.com/k2l4Hx1YL8lnyzId9JX-ZnJsY_YEVoL6M5Yuialv-BM/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9oYWNr/ZXJub29uLmltZ2l4/Lm5ldC9pbWFnZXMv/clhkS1diWks3TmZY/S1JYV1R6S1owb3JL/ZnZxMS1sNzkzcHVu/LmpwZWc_YXV0bz1m/b3JtYXQmZml0PW1h/eCZ3PTM4NDA",
        width=100,
    )
    st.markdown(
        """
    **Features:**
    - Summarize YouTube Videos
    - Summarize Website Content
    """
    )
    add_vertical_space(2)
    st.markdown(
        """
    **Instructions:**
    1. Enter your Groq API key.
    2. Paste a valid YouTube or website URL.
    3. Click "Summarize" to generate a summary.
    """
    )
    add_vertical_space(2)
    groq_api_key = st.text_input("Groq API Key", value="", type="password")
    st.markdown("---")
    st.markdown("Built by Jayanth Nayak (https://www.linkedin.com/in/jayanthnayak373/)")

### Main App Content
st.title("🌟 LangChain: Summarize Text From YouTube/Website 🌐")
st.markdown(
    """
Welcome to the **LangChain Summarizer**! Use this app to generate concise and insightful summaries from YouTube videos or website content. 🚀
"""
)

### Input Section
st.subheader("Enter the URL to Summarize")
generic_url = st.text_input(
    "Paste YouTube or Website URL here 👇",
    placeholder="https://youtube.com/watch?v=example",
    label_visibility="visible",
)

### Prompts
initial_prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
initial_prompt = PromptTemplate(template=initial_prompt_template, input_variables=["text"])

refine_prompt_template = """
We have an existing summary: {existing_answer}
Refine the summary with the following additional content:
Content: {text}

Provide a concise and improved version of the summary.
"""
refine_prompt = PromptTemplate(template=refine_prompt_template, input_variables=["existing_answer", "text"])

### Summarization Button
if st.button("🚀 Summarize Content"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("❌ Please provide both the Groq API key and a valid URL.")
    elif not validators.url(generic_url):
        st.error("❌ Invalid URL. Please enter a valid YouTube or website URL.")
    else:
        try:
            with st.spinner("⏳ Processing..."):
                docs = []
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        with YoutubeDL({"quiet": True}) as ydl:
                            info = ydl.extract_info(generic_url, download=False)
                            video_description = info.get("description", "")
                            video_title = info.get("title", "")
                            docs = [Document(page_content=f"Title: {video_title}\n\n{video_description}")]
                    except Exception as yt_error:
                        st.error(f"⚠️ Failed to load YouTube video: {yt_error}")
                else:
                    try:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": (
                                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 "
                                    "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                                )
                            },
                        )
                        raw_docs = loader.load()
                        docs = [Document(page_content=doc.page_content) for doc in raw_docs]
                    except Exception as url_error:
                        st.error(f"⚠️ Failed to load URL: {url_error}")

                if docs:
                    llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
                    chain = load_summarize_chain(
                        llm,
                        chain_type="refine",
                        question_prompt=initial_prompt,
                        refine_prompt=refine_prompt,
                    )
                    output_summary = chain.run(docs)
                    st.success("🎉 Summary Generated!")
                    st.text_area("Generated Summary", value=output_summary, height=300)
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")

### Footer
st.markdown("---")

