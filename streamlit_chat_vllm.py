import streamlit as st
from htmlTemplates import css


import requests
import json 

#Langchain and vLLM libraries
from langchain_community.llms import VLLM

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

"""
Aim: make an interactive chat using the vllm chatbot Mistral

"""
@st.cache_resource
def load_chain():
    template = """you are an Artificial Intelligence agent and answer the following question carefully
    Question: {question}

    Answer: """
    prompt = PromptTemplate.from_template(template)


    #Step 2: creating the vllm model instance
    vllm_kwargs = {}
    vllm_kwargs['max_model_len'] = 20000
    llm = VLLM(
        model="TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
        trust_remote_code=True,  # mandatory for hf models
        top_k=10,
        top_p=0.95,
        temperature=0.8,
        dtype='half',
        vllm_kwargs=vllm_kwargs
        
    )
    #Step 3: making the LLM chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    st.success("Loaded Mistral model successfully!") 
    return llm_chain



def main():
    
    st.set_page_config(page_title="Chat with our Agent!",
                       page_icon=":books:")
    #st.write(css, unsafe_allow_html=True)

    #Step 1: creating the prompt template

    

    
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]


    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def generate_response(user_question):
        #here we will write the user question and the password
        
        #Here we need to infer the model through vllm and get the response back
        llm_chain1 = load_chain()
        ai_answer = llm_chain1.invoke(user_question)
        #st.write(response)
        #Now we got the response from the ai agent and now we will return the response
        #we will return just the value of the answer
        return (ai_answer)
            
    # User-provided prompt
    if user_question := st.chat_input("Ask general questions"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)
    
    #now we will generate the response if the last message is not the ai's chat
    if st.session_state.messages[-1]["role"] != "assistant":
        #now we will generate the response
        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
                response_ai = generate_response(user_question)
                st.write(response_ai)

        new_message = {"role": "assistant", "content": response_ai}
        st.session_state.messages.append(new_message)






if __name__ == '__main__':
    main()