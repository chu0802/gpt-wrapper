# GPT-Wrapper

Every time I need to use ChatGPT's API, I need to write the same code, and this is annoying. Thus, I write this wrapper.

This is a Python wrapper for OpenAI's GPT models. It is easy to use for research and development.

## Features

- Multimodal Support: Send both text and images to GPT models like GPT-4o
- Structured Responses: Get typed, structured responses using Pydantic models
- Cost Tracking: Monitor token usage and associated costs
- Simple API: Clean, intuitive interface for sending prompts and handling responses

## Installation

```bash
# Clone the repository
pip install -r requirements.txt
```

## Usage

Here is an example of how to use GPT-4o to recognize objects in an image:

1. Create a `.env.sh` file with your OpenAI credentials:

```bash
API_KEY=your_openai_api_key
ORGANIZATION=your_openai_organization_id
```

2. Source the `.env.sh` file:

```bash
source .env.sh
```

3. Use the wrapper:

```python
from src.api import GPTWrapper

# Initialize wrapper
gpt_wrapper = GPTWrapper(model_name="gpt-4o")

# Send a query with an image
result = gpt_wrapper.ask(
    image="path/to/image.png", 
    text="What is this?"
)

# Access structured result
print(result)

# Display cost information
gpt_wrapper.show_cost()
```

## Customization

### Creating Custom Response Models


```python
from pydantic import BaseModel

class DetailedObjectResponse(BaseModel):
    name: str
    ... # other fields that you want to include in the response
```

### Custom System Prompts

Modify the system prompt in `src/prompts.py` to customize the behavior:

```python
SYSTEM_PROMPT = "Analyze images and provide detailed information about objects."
```

## Cost Tracking
The wrapper automatically tracks token usage and costs. Supported models:

- gpt-4o: $0.0025 per 1K prompt tokens, $0.01 per 1K completion tokens
- gpt-4o-mini: $0.00015 per 1K prompt tokens, $0.0006 per 1K completion tokens

Access cost information:

```python
gpt_wrapper.show_cost()
```

License
This project is licensed under the MIT License - see the LICENSE file for details.
