from gptwrapper import GPTWrapper
from gptwrapper.response import ObjectRecognitionResponse, InteractionResponse
from gptwrapper.config.system_prompt import RECOGNITION_SYSTEM_PROMPT, INTERACTION_SYSTEM_PROMPT

RECOG_PROMPT = "What is this?"
INTERACTION_PROMPT = "show me 5 different interactions to interact with {object_name} that can produce unique and interesting sounds. The format should contain only one action + the object name. e.g., opening the door"

def main():
    gpt = GPTWrapper(model_name="gemini-2.5-flash-preview-04-17")

    recog_result = gpt.ask(
        image="examples/kettle.png", 
        text=RECOG_PROMPT,
        system_message=RECOGNITION_SYSTEM_PROMPT,
        response_format=ObjectRecognitionResponse,
    )
    
    object_name = recog_result.name

    interaction_result = gpt.ask(
        text=INTERACTION_PROMPT.format(object_name=object_name),
        system_message=INTERACTION_SYSTEM_PROMPT,
        response_format=InteractionResponse,
    )

    print(f"object name: {object_name}")
    print(f"interaction result: {interaction_result}")
    print(f"total cost: {gpt.show_cost()}")

if __name__ == "__main__":
    main()
