from vllm import LLM, SamplingParams

from constants import model_llama

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

llm = LLM(model="TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
        quantization='awq', max_model_len=20000,  dtype='half'
          )

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")