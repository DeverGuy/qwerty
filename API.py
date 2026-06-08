import os
import base64
# pyrefly: ignore [missing-import]
from openai import OpenAI

# Initialize the OpenAI client with NVIDIA's API URL
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("NVIDIA_API_KEY", "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")
)

# 1. Path to your image (we'll use one of your existing images as a test)
image_path = "Face_recognition_-With-Emotions-/test_face.jpg"

# 2. Read and Base64 encode the image
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# 3. Call the Vision model
print(f"Sending image '{image_path}' to the API...")
completion = client.chat.completions.create(
  model="meta/llama-3.2-90b-vision-instruct",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image? Describe it in detail."},
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  temperature=0.6,
  top_p=0.95,
  max_tokens=1024,
  stream=False
)

# 4. Print the result
print("\n--- Output from Model ---")
print(completion.choices[0].message.content)
