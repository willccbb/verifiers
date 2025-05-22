from copy import deepcopy
from typing import Dict, Any 
from openai import OpenAI
client = OpenAI()


sampling_args = {'extra_body': {
    'skip_special_tokens': False,
    'spaces_between_special_tokens': False,
}}

def sanitize_sampling_args(client: OpenAI, sampling_args: Dict[str, Any]) -> Dict[str, Any]:
    from urllib.parse import urlparse
    url = urlparse(str(client.base_url))
    # check if url is not localhost/127.0.0.1/0.0.0.0
    if url.netloc not in ["localhost", "127.0.0.1", "0.0.0.0"]:
        sanitized_args = deepcopy(sampling_args)
        # remove extra_body
        sanitized_args.pop('extra_body', None)
        return sanitized_args
    return sampling_args


completion = client.chat.completions.create(
  model="gpt-4.1",
  messages=[
      {
          "role": "user",
          "content": "Write a one-sentence bedtime story about a unicorn."
      }
  ],
  **sanitize_sampling_args(client=client,sampling_args=sampling_args)
)

print(completion.choices[0].message.content)

