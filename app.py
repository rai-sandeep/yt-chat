import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

def get_documents(video_links):
    loader = YoutubeLoader.from_youtube_url(video_links, add_video_info=True)
    return loader.load_and_split()


def get_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.2)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

def print_chat():
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with YouTube Videos",
                       page_icon=":video_camera:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with YouTube Videos :video_camera:")

    chat_container = st.container()    
                
    user_question = st.text_input("Ask a question about your YouTube Videos:")
    if user_question:
        handle_userinput(user_question)
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    print(message.content)
                    st.write(bot_template.replace(
                        "{{MSG}}", message.content.replace("\n", "<br>")), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your videos")
        video_links = st.text_input("Enter your YouTube Video links here and click on 'Process'")
        if st.button("Process"):
            with st.spinner("Processing"):
                # get documents
                documents = get_documents(video_links)

                # create vector store
                vectorstore = get_vectorstore(documents)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()