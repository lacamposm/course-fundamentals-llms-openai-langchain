import os
import logging
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"


def chatbot_template(system_message, user_question):
    """
    Create a template to pass custom queries to LLM 
    :param user_question: Message user.
    :param system_message: Message to the system, like a rol.
    :return: Configured with the system message, chat history, and user question.
    """
    chat_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(user_question)
        ]
    )
    return chat_prompt


def stream_response_with_memory_openai(model_name, query, chat_history):
    """
    Streams a response for a given query using an OpenAI model, formatted to maintain conversation history.

    :param model_name: The name of the OpenAI model to use for generating responses.
    :param query: The current user question or input.
    :param chat_history: The history of previous interactions to be considered for context.
    :return: A streaming object that continuously provides the chatbot's responses.
    """
    system_prompt = """You are a friendly chatbot having a conversation with a human and giving answers only in \
    Spanish."""
    user_question = "{user_question}"
    prompt = chatbot_template(system_message=system_prompt, user_question=user_question)

    chat_openai = ChatOpenAI(
        model=model_name,
        temperature=0.1,
    )

    chain = prompt | chat_openai | StrOutputParser()

    return chain.stream(input={"chat_history": chat_history, "user_question": query})


# Interfaz de usuario con streamlit #
st.title("🦜 Chatbot - OIC 🦜")
with st.sidebar.expander("Modelos de OPENAI"):
    model_openai = st.radio(
        "Seleccione su ChatModel:",
        ["gpt-3.5-turbo-0125", "gpt-4-0125-preview"],
        key="openai_chat_model"
    )
    st.session_state["openai_model"] = model_openai

# Iniciamos un chat history #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

user_query = st.chat_input("Your message")

if user_query is not None and user_query != "":
    logging.info("Inicio del chat")
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        llm_response = st.write_stream(
            stream_response_with_memory_openai(
                model_name=st.session_state["openai_model"],
                query=user_query,
                chat_history=st.session_state.chat_history)
        )
    
    st.session_state.chat_history.append(AIMessage(llm_response))
    logging.info("Fin del chat.\n==============================================================================")
