import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_documents(video_links):
    loader = YoutubeLoader.from_youtube_url(video_links, add_video_info=True)
    return loader.load_and_split()


def get_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    st.session_state.chat_history = None
    
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

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with YouTube Video",
                       page_icon=":video_camera:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "docmeta" not in st.session_state:
        st.session_state.docmeta = None

    st.header("Chat with YouTube Video :video_camera:")

    video_container = st.container() 
    chat_container = st.container()

    # Write empty text to chat container to stop duplication
    # https://github.com/streamlit/streamlit/issues/4652
    chat_container.write("")  

    with video_container:
        st.subheader("Your video")
        video_links = st.text_input("Enter your YouTube Video link here and click on 'Process'")
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # get documents
                documents = get_documents(video_links)

                # create vector store
                vectorstore = get_vectorstore(documents)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                
                st.session_state.docmeta = documents[0].metadata

        if (st.session_state.docmeta):
            st.write("Title: "+st.session_state.docmeta["title"])
            st.write("Author: "+st.session_state.docmeta["author"])
            st.image(st.session_state.docmeta["thumbnail_url"])
            st.write("")

    with st.form(key='chat_form', clear_on_submit=True):    
        user_question = st.text_input("Ask a question about your YouTube Video:")
        ask_button = st.form_submit_button(label='Ask')

    if ask_button and user_question:
        with st.spinner("Asking"):
            handle_userinput(user_question)
            with chat_container:
                for i, msg in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        message(msg.content, is_user=True, key=str(i) + '_user')
                    else:
                        message(msg.content, key=str(i))


if __name__ == '__main__':
    main()