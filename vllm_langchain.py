from langchain_community.llms import VLLM


"""
Aim: Demonstrating the basic inferencing
this code is for the using the langchain module for developing the vllm inferencing
"""


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

print(llm.invoke("The capital of France is"))