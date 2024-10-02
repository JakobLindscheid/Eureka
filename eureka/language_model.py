import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from huggingface_hub import InferenceClient
from joblib import Parallel, delayed
import google.generativeai as genai
from groq import Groq
import openai

genai.configure(api_key=os.getenv("GEMINI_TOKEN"))
openai.api_key = os.getenv("OPENAI_API_KEY")

class LanguageModel:

    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", provider="local"):
        self.provider = provider
        self.model_name = model_name
        if self.provider == "huggingface":
            self.client = InferenceClient(model_name,token=os.getenv("HF_TOKEN"),headers={"x-use-cache":"false"})
        
        elif self.provider == "gemini":
            self.model = genai.GenerativeModel(model_name)

        elif self.provider == "groq":
            self.client = Groq(api_key=os.getenv("GROQ_KEY"))
        
        elif self.provider == "local":
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True, padding_side="left", maximum_length = 2048, model_max_length = 2048, token=HF_TOKEN)
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map = 'auto', token=os.getenv("HF_TOKEN"))
            """ tokenizer.pad_token = tokenizer.eos_token
            model.generation_config.pad_token_id = model.generation_config.eos_token_id """
            self.generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    def generate_text(self, prompt, temperature, n):        
        if self.provider == "huggingface":
            def api_call(_):
                return self.client.chat_completion(
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=4096,
                    stream=False,
                ).choices[0].message.content
            results = Parallel(n_jobs=-1)(delayed(api_call)(_) for _ in range(n))

        elif self.provider == "gemini":
            messages = []
            for i in range(len(prompt)):
                if prompt[i]["role"] == "system":
                    self.model = genai.GenerativeModel("gemini-1.5-flash-latest", system_instruction=prompt[i]["content"])
                elif prompt[i]["role"] == "user":
                    messages.append({"role":"user", "parts":prompt[i]["content"]})
                elif prompt[i]["role"] == "assistant":
                    messages.append({"role":"model", "parts":prompt[i]["content"]})

            def api_call(_):
                return self.model.generate_content(contents=messages, generation_config=genai.GenerationConfig(temperature=temperature)).text
            results = Parallel(n_jobs=-1)(delayed(api_call)(_) for _ in range(n))

        elif self.provider == "groq":
            def api_call(_):
                return self.client.chat.completions.create(
                                        messages=prompt,
                                        model=self.model_name,
                                        temperature=temperature
                    ).choices[0].message.content
            # results = Parallel(n_jobs=-1)(delayed(api_call)(_) for _ in range(n))
            results = [api_call(_) for _ in range(n)]

        elif self.provider == "local":
            results = self.generator([prompt for _ in range(n)], max_length=4096, temperature=temperature, truncation=True, return_full_text=False)
            results = [result[0]["generated_text"] for result in results]

        elif self.provider == "openai":
            return openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        n=n
                    )
        return {"choices": [{"message":{"content":results[i]}} for i in range(n)]}