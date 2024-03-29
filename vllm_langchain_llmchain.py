from langchain_community.llms import VLLM

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

"""
Aim: 

"""

#Step 1: creating the prompt template
template = """Question: {question}

Answer: Let's think step by step."""
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

#print(llm.invoke("The capital of France is"))


#Step 3: making the LLM chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who was the US president in the year the first Pokemon game was released?"

print(llm_chain.invoke(question))