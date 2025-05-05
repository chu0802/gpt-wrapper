from src.api import GPTWrapper

def main():
    gpt_wrapper = GPTWrapper(model_name="gpt-4o")

    result = gpt_wrapper.ask(image="examples/kettle.png", text="What is this?")
    
    print(result)
    gpt_wrapper.show_cost()

if __name__ == "__main__":
    main()
