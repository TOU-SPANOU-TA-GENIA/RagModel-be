import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
import logging

logger = logging.getLogger(__name__)

class LLMProvider:
    def __init__(self):
        # Update this to your specific model path
        self.model_name = "./offline_models/qwen3-4b" 
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"🚀 [LLM] Loading model: {self.model_name}...")
        
        # 4-bit Quantization for Speed and Memory Efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        # Enable KV caching for faster inference
        self.model.config.use_cache = True
        print("✅ [LLM] Model loaded with 4-bit quantization.")

    def stream(self, messages: list):
        """
        Generates a stream of tokens from the model.
        """
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Run generation in a separate thread to allow streaming
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text