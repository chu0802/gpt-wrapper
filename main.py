from src.api import GPTWrapper
from src.response import ObjectRecognitionResponse

def main():
    gpt_wrapper = GPTWrapper(model_name="gpt-4o", response_format=ObjectRecognitionResponse)

    result = gpt_wrapper.ask(image="examples/kettle.png", text="What is this?")
    
    print(result)
    gpt_wrapper.show_cost()

if __name__ == "__main__":
    main()
