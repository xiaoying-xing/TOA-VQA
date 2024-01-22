import openai

def set_openai_key(key):
    openai.api_key = key

def call_chatgpt(chatgpt_messages, max_tokens=100, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=0.1,
                                            max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

def call_gpt3(messages, max_tokens=100, model="text-davinci-003"):
    response = openai.Completion.create(model=model, prompt=messages, max_tokens=max_tokens, temperature=0.2)  # temperature=0.6,
    reply = response['choices'][0]['text']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens