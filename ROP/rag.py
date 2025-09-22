import yaml
import os
from pathlib import Path
import re
import time
import base64
import random
from openai import OpenAI
from anthropic import Anthropic
import json
from PIL import Image
from io import BytesIO




def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')
        
class RAG:
    def __init__(self, config_file):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found.")
        try:
            with open(config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
            
        # if 'keyword_extraction' not in config:
        #     raise KeyError("Missing required key 'rag' in config file.")

        self.params = config.get('rag', {})            
        self.api_to_use = config.get('use_api', "")
        
        # Initialize OpenAI instance
        if self.api_to_use == 'openai':
            openai_api_key = config.get('openai_api_key', None)
            if openai_api_key:
                self.llm = OpenAI(api_key = openai_api_key)
                print('Connected to OpenAI API for RAG...')
            else:
                print('Warning: OpenAI API key is not provided!')

        elif self.api_to_use == 'claude':
            # Initialize Claude (Anthropic) instance
            claude_api_key = config.get('claude_api_key', None)
            if claude_api_key:
                self.llm = Anthropic(api_key=claude_api_key)
                print('Connected to Claude API for RAG...')
            else:
                print('Warning: Claude API key is not provided!')
                
        else:
            print(f'The api {self.api_to_use} is not supported!')
    


    def summarize_image(self, image_path):
        if self.api_to_use != "openai":
            print("This method only supports GPT-4o via OpenAI.")
            return

        # image_base64 = image_to_base64(image_path)
        prompt = self.params.get('prompt_summarize', "")

        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": prompt},
        #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        #         ]
        #     }
        # ]

        orig = Image.open(image_path)


# Convert it to RGB
        rgb_image = orig.convert("RGB")
        rgb_buf = BytesIO()
        rgb_image.save(rgb_buf, format="PNG") 
        rgb_buf.name = "image.png"
        rgb_buf.seek(0)

        l_buf = BytesIO()
        mask_img = orig.convert("L")
        mask_img.save(l_buf, format="PNG")
        l_buf.name = "mask.png"
        l_buf.seek(0)

        response = self.llm.images.edit(
            image=rgb_buf,
            mask=l_buf,
            prompt=prompt,
            n=1,
            size="256x256",
            response_format="b64_json"
        )
        image_data = response.data[0].b64_json
        return image_data
if __name__ == "__main__":
    # Example usage
    # Ensure you have a config.yaml file with the required keys

    myRAG = RAG("/media/yuganlab/blackstone/Sarah/config.yaml")
    root = "/media/yuganlab/blackstone/HZL/Data/synthetic"
    files = os.listdir(root)
    output_path = "/media/yuganlab/blackstone/Sarah/gpt_images"
    for sub in files:
        subp = Path(root) / sub
        if not subp.is_dir(): continue
        for img in os.listdir(subp):
            image_path = Path(subp) / img
            print(f"Processing {image_path}...")
            result = myRAG.summarize_image(image_path)
            img_bytes = base64.b64decode(result)
            buffer = BytesIO(img_bytes)
            img = Image.open(buffer)
            print(f"Saved result to {output_path}")
            img.save(f"{output_path}/gen_{img}.png")
        

