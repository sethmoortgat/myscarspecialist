from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from src.prompts import (
    template_text_system,
    context_template_text,
)

from src.services import (
    get_context,
    get_llm_response,
    transfor_user_question,
)

import streamlit as st
import hmac
import logging

# configure logger
logging.basicConfig(
    level=logging.INFO,  # Logniveau (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Logformaat
)


def main():
    # ***************
    #
    # PASSWORD
    #
    # ***************

    st.set_page_config(
        layout="centered", page_icon="./data/SCARBOT_AVATAR.png", page_title="Scarbot"
    )  # centered wide

    def check_password():
        """Returns `True` if the user had the correct password."""

        # Check if check_password is set to False in secrets.toml
        if "check_password" in st.secrets and st.secrets["check_password"] == False:
            return True

        # If check_password setting is missing or True, proceed with password check
        def password_entered():
            """Checks whether a password entered by the user is correct."""
            if hmac.compare_digest(
                st.session_state["password"], st.secrets["password"]
            ):
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store the password.
            else:
                st.session_state["password_correct"] = False

        # Return True if the passward is validated.
        if st.session_state.get("password_correct", False):
            return True

        # Show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        if "password_correct" in st.session_state:
            st.error("😕 Password incorrect")
        return False

    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

    # ***************
    #
    # utility functions
    # within streamlit app
    #
    # ***************

    def new_question():
        st.session_state.new_question = True
        st.session_state.question = ""
        st.session_state.context = ""
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.n_questions = 0
        logging.info("New quesiton started")

    def existing_question():
        st.session_state.new_question = False
        logging.info(f"New question: {st.session_state.question}")

    def add_user_input():
        st.session_state.messages.append(
            {"role": "user", "content": st.session_state.chat_input}
        )
        st.session_state.chat_history.append(
            {"role": "user", "content": st.session_state.chat_input}
        )
        logging.info(f"New question: {st.session_state.chat_input}")

    # st.set_page_config(
    #         layout="centered",
    #         page_icon="./data/SCARBOT_AVATAR.png",
    #         page_title="Scarbot"
    #     )  # centered wide

    @st.cache_resource
    def get_qdrant_client():
        if "PATH_TO_VECTORSTORE" not in st.secrets:
            st.error("Error: PATH_TO_VECTORSTORE is not defined in secrets.toml")
            st.stop()
        vs_path = st.secrets["PATH_TO_VECTORSTORE"]
        client = QdrantClient(path=vs_path)
        return client

    # ***************
    #
    # initialize the session state
    #
    # ***************

    if "language" not in st.session_state.keys():
        # for example when you browse to http://localhost:8501/?lang=nl, lang will be read as nl
        query_params = st.query_params
        url_language = query_params.get("lang", "EN").upper()
        if url_language in ["NL", "EN"]:
            st.session_state.language = url_language
        else:
            st.session_state.language = "EN"
        logging.info(f"Language initialized from URL: {st.session_state.language}")

    if "question" not in st.session_state.keys():
        st.session_state.question = ""

    if "context" not in st.session_state.keys():
        st.session_state.context = ""

    if "new_question" not in st.session_state.keys():
        st.session_state.new_question = True

    if "embedding_function" not in st.session_state.keys():
        st.session_state.embedding_function = OpenAIEmbeddings(
            api_key=st.secrets["openai_api_key"],
            model="text-embedding-3-large",
        )

    if "vectorstore" not in st.session_state.keys():
        st.session_state.vectorstore = QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name="myscarspecialist",
            embedding=st.session_state.embedding_function,
        )

    if "openai_client" not in st.session_state.keys():
        st.session_state.openai_client = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=st.secrets["openai_api_key"],
        )

    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    if "chat_history" not in st.session_state.keys():
        st.session_state.chat_history = []

    # ***************
    #
    # Landing page for new question
    #
    # ***************

    if st.session_state.new_question:
        col1, col2, col3, col4 = st.columns([1, 3, 10, 1])
        with col2:
            st.image("./data/SCARBOT_AVATAR.png", width=200)
        with col3:
            st.markdown(
                "## Hi, my name is Scarbot!"
                if st.session_state.language == "EN"
                else "## Hallo, mijn naam is Scarbot!"
            )
            st.markdown(
                "## How can I help you?"
                if st.session_state.language == "EN"
                else "## Hoe kan ik je helpen?"
            )

        label = (
            "Ask your question"
            if st.session_state.language == "EN"
            else "Stel je vraag"
        )

        st.text_input(
            "original question",
            placeholder=label,
            key="question",
            on_change=existing_question,
            label_visibility="collapsed",
        )

    # ***************
    #
    # Chat continuation
    #
    # ***************

    else:
        if len(st.session_state.messages) == 0:
            # First initialise the system prompt
            system_prompt = template_text_system.format(
                language="dutch" if st.session_state.language == "NL" else "english"
            )
            st.session_state.messages.append(
                {"role": "system", "content": system_prompt}
            )
            st.session_state.chat_history.append(
                {"role": "system", "content": system_prompt}
            )

            # Then add the context prompt
            with st.spinner(
                "Browsing website..."
                if st.session_state.language == "EN"
                else "Website doorzoeken..."
            ):
                st.session_state.context = get_context(
                    st.session_state.question,
                    st.session_state.vectorstore,
                    n_chunks=3,
                    filters=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.language",
                                match=models.MatchValue(
                                    value="nl"
                                    if st.session_state.language == "NL"
                                    else "en"
                                ),
                            ),
                        ]
                    ),
                )
            context_prompt = context_template_text.format(
                context=st.session_state.context
            )
            st.session_state.messages.append(
                {"role": "system", "content": context_prompt}
            )
            st.session_state.chat_history.append(
                {"role": "system", "content": context_prompt}
            )

            # Finally add the user question
            st.session_state.messages.append(
                {"role": "user", "content": st.session_state.question}
            )
            st.session_state.chat_history.append(
                {"role": "user", "content": st.session_state.question}
            )

        else:
            last_question = next(
                (
                    msg["content"]
                    for msg in reversed(st.session_state.messages)
                    if msg["role"] == "user"
                ),
                None,
            )
            if last_question:
                transformed_last_question = transfor_user_question(
                    last_question,
                    st.session_state.messages,
                    st.session_state.openai_client,
                )
                logging.info(
                    f"User question was expanded for RAG to: {transformed_last_question}"
                )
            else:
                transformed_last_question = ""
                logging.error(
                    "There was no last user question found in the chat history!"
                )

            with st.spinner(
                "Browsing website..."
                if st.session_state.language == "EN"
                else "Website doorzoeken..."
            ):
                follow_up_context = get_context(
                    transformed_last_question,
                    st.session_state.vectorstore,
                    n_chunks=2,
                    filters=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.language",
                                match=models.MatchValue(
                                    value="nl"
                                    if st.session_state.language == "NL"
                                    else "en"
                                ),
                            ),
                        ]
                    ),
                )
            context_prompt = context_template_text.format(context=follow_up_context)
            st.session_state.messages.append(
                {"role": "system", "content": context_prompt}
            )
            st.session_state.chat_history.append(
                {"role": "system", "content": context_prompt}
            )

        with st.spinner(
            "Composing answer..."
            if st.session_state.language == "EN"
            else "Antwoord genereren..."
        ):
            answer = get_llm_response(
                st.session_state.messages,
                st.session_state.openai_client,
            )
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # show the chat history on screen
        with st.container():
            for idx, message in enumerate(st.session_state.chat_history):
                if message["role"] == "system":
                    continue
                else:
                    if message["role"] == "assistant":
                        icon = "./data/avatar_icon.png"
                    else:
                        icon = "./data/user_icon.png"
                    with st.chat_message(message["role"], avatar=icon):
                        st.write(message["content"])

        col1, col2 = st.columns([3, 1])
        with col1:
            st.chat_input(
                "Enter your follow-up question..."
                if st.session_state.language == "EN"
                else "Zet het gesprek verder...",
                key="chat_input",
                on_submit=add_user_input,
            )
        with col2:
            st.button(
                "Clear chat history"
                if st.session_state.language == "EN"
                else "Chatgeschiedenis wissen",
                on_click=new_question,
                type="secondary",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
