import os

# 1. Attempt to load the token securely based on the environment
try:
    # Try Google Colab Secrets first
    from google.colab import userdata
    HF_TOKEN = userdata.get('generativeAI')
except ImportError:
    # Fallback to local environment variables if running outside of Colab
    # Note for local users: requires `pip install python-dotenv` and a local .env file
    from dotenv import load_dotenv
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

# Ensure the token was successfully loaded before proceeding
assert HF_TOKEN, "ERROR: Token missing. Set it in Colab Secrets ('generativeAI') or locally in a .env file ('HF_TOKEN')."

# 2. Initialize the Hugging Face client
from huggingface_hub import InferenceClient
client = InferenceClient(api_key=HF_TOKEN)

# Select the General Purpose curie model for text
model = "meta-llama/Meta-Llama-3-8B-Instruct"

text_prompt1="A very long time ago, there was a small village on the edge of the great forest. This was a peaceful village most of the time, but the villagers lived in fear of the Lobizon, who were said to dwell deep within the forest. The Lobizon were dark creatures, half man and half wolf, and every full moon it was said that these creatures would creep out of the forest in search of human flesh." 
text_prompt2="But how does such a creature come into being? That is simple: the curse upon the seventh son born to any family. The curse will not befall any daughters, but if a mother gives birth to seven sons, then the last of these sons will surely become a Lobizon." 

text_prompt3="When Filipe was born his mother was afraid. She had hoped for a daughter, not a seventh son; but Filipe’s mother was kind and loving and she was not going to turn her back on her own child, no matter what the villagers said about the curse. tl;dr:"

text_block = text_prompt1 + text_prompt2 + text_prompt3
response = client.chat_completion(
  model=model,
  messages = [{"role":"system", "content":"You are a helpful assistant."},
               {"role":"user","content":text_block}],
  max_tokens=250
  )

print(response.choices[0].message.content)
