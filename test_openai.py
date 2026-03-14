import os
from openai import OpenAI

key = os.environ.get("OPENAI_API_KEY")
print("KEY FOUND:", bool(key))
print("KEY PREFIX:", key[:12] if key else "NONE")

client = OpenAI(api_key=key)

try:
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0
    )
    print("SUCCESS")
    print(r.choices[0].message.content)
except Exception as e:
    print("ERROR:", e)