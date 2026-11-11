import os, time
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(".env")
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"])
MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
USER = {"role": "user", "content": "Reply with the JSON {\"ok\": true} and nothing else.", "chat_template_kwargs": {"enable_thinking": False}}

resp = client.chat.completions.create(
    model=MODEL,
    messages=[USER],
    max_tokens=200,)

print(resp.choices[0].message.content)