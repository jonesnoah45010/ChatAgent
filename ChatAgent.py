import os
from openai import OpenAI
import asyncio
import tiktoken
import pickle





def save_ChatAgent_as_txt(agent,filename):
    if ".txt" not in filename:
        filename = filename+".txt"
    with open(filename, 'w') as myfile: 
        content = repr(agent)
        myfile.write(content)

def load_ChatAgent_from_txt(filename):
    repr_str=None
    with open(filename, 'r') as myfile: 
        repr_str = myfile.read()
    new_instance = eval(repr_str)
    return new_instance


def save_ChatAgent_as_pickle(agent, filename):
    """
    Save a ChatAgent instance to a pickle file, excluding non-pickleable objects like the OpenAI client.
    """
    temp_client = agent.client  # Temporarily store the OpenAI client
    agent.client = None  # Remove the client before pickling
    if ".pickle" not in filename:
        filename = filename + ".pickle"
    with open(filename, 'wb') as file:
        pickle.dump(agent, file)
    agent.client = temp_client  # Restore the client after pickling


def load_ChatAgent_from_pickle(filename, api_key):
    """
    Load a ChatAgent instance from a pickle file and reinitialize non-pickleable objects like the OpenAI client.
    """
    with open(filename, 'rb') as file:
        agent = pickle.load(file)
    # Reinitialize the OpenAI client if the API key is available
    agent.client = OpenAI(api_key=api_key)
    return agent











class ChatAgent:
    def __init__(self, api_key=None, model="gpt-3.5-turbo", messages=None, token_limit=4096, summary_size=300):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.messages = messages if messages is not None else []
        self.token_limit = token_limit
        self.enc = tiktoken.encoding_for_model(self.model)
        self.primary_directive = None
        self.summary_size = summary_size


    def set_primary_directive(self, system_prompt=None):
        if system_prompt is None and self.primary_directive is not None:
            system_prompt = self.primary_directive
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
            self.primary_directive = system_prompt

    def count_tokens(self):
        num_tokens = 0
        for message in self.messages:
            num_tokens += len(self.enc.encode(message["content"]))
        return num_tokens

    def is_within_token_limit(self, token_limit=None):
        if token_limit is None:
            token_limit = self.token_limit
        current_token_count = self.count_tokens()
        return current_token_count <= token_limit

    def tokens_left(self):
        t = self.count_tokens()
        return int(self.token_limit) - int(t)

    def add_context(self, system_prompt=None):
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def refresh_session(self):
        summary = self.summarize_current_conversation()
        self.messages = []
        self.set_primary_directive()
        self.add_context("In a previous conversation, the following was discussed ... " + str(summary))


    def send_message(self, user_message):
        tokens_used_user = len(self.enc.encode(user_message))
        current_token_count = self.count_tokens()
        print("CURRENT TOKENS USED: " + str(current_token_count))
        print("MAX TOKENS: " + str(self.token_limit))
        if current_token_count + tokens_used_user > self.token_limit:
            print("ABOUT TO GO OVER TOKEN LIMIT")
            self.refresh_session()
            print("CONVERSATION WAS REFRESHED")

        # Add user message to conversation history
        self.messages.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(model=self.model,
                                                       messages=self.messages)
        ai_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": ai_message})
        return ai_message

    async def send_message_async(self, user_message):
        # Use asyncio.to_thread to run the synchronous side_message method asynchronously
        response = await asyncio.to_thread(lambda: self.send_message(user_message))
        return response  # side_message already returns the content

    def get_conversation_history(self):
        # Returns the entire conversation history
        return self.messages


    def side_message(self, prompt, use_context = False):
        # get side message that will not affect overall conversation or be added to conversation history
        temp_messages = []
        if use_context:
            temp_messages = self.messages.copy()
        temp_messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(model=self.model,
                                                       messages=temp_messages)
        return response.choices[0].message.content


    async def side_message_async(self, prompt, use_context=False):
        # Use asyncio.to_thread to run the synchronous side_message method asynchronously
        response = await asyncio.to_thread(lambda: self.side_message(prompt, use_context))
        return response  # side_message already returns the content


    def summarize_current_conversation(self,in_max_n_words=None):
        if in_max_n_words is None:
            in_max_n_words = self.summary_size
        q = """
        Summarize the entire conversation we have had in in_max_n_words
        words or less. Make a note of key information you learned about 
        the user and what key recomendations you gave to the user or 
        key information you shared with the user.
        """
        q = q.replace("in_max_n_words",str(in_max_n_words))
        return self.side_message(q, use_context=True)


    def __repr__(self):
        repr_str = (
            f"ChatAgent(api_key='{self.api_key}', "
            f"model='{self.model}', "
            f"token_limit={self.token_limit}, "
            f"messages={repr(self.messages)}, "
            f"summary_size={repr(self.summary_size)})"
        )
        return repr_str

    def repr_no_key(self):
        repr_str = (
            f"ChatAgent(api_key=None, "
            f"model='{self.model}', "
            f"token_limit={self.token_limit}, "
            f"messages={repr(self.messages)}, "
            f"summary_size={repr(self.summary_size)})"
        )
        return repr_str

    def save_as_txt(self, filename):
        if ".txt" not in filename:
            filename = filename+".txt"
        with open(filename, 'w') as myfile: 
            content = repr(self)
            myfile.write(content)

    def load_from_txt(self, filename):
        repr_str=None
        with open(filename, 'r') as myfile: 
            repr_str = myfile.read()
        new_instance = eval(repr_str)
        attributes = [
            attr
            for attr in dir(new_instance)
            if not callable(getattr(new_instance, attr)) and not attr.startswith("__")
        ]
        for attr in attributes:
            setattr(self, attr, getattr(new_instance, attr))

    def save_as_pickle(self, filename):
        temp_client = self.client  # Temporarily store the OpenAI client
        self.client = None  # Remove the client before pickling
        if ".pickle" not in filename:
            filename = filename + ".pickle"
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        self.client = temp_client  # Restore the client after pickling

    def load_from_pickle(self, filename):
        with open(filename, 'rb') as file:
            loaded_instance = pickle.load(file)
        # Update current instance's attributes (excluding special methods)
        attributes = [
            attr
            for attr in dir(loaded_instance)
            if not callable(getattr(loaded_instance, attr)) and not attr.startswith("__")
        ]
        for attr in attributes:
            setattr(self, attr, getattr(loaded_instance, attr))
        # Reinitialize the OpenAI client
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)







if __name__ == "__main__":

    OPENAI_API_KEY = "YOUR_OPEN_AI_API_KEY"

    agent = ChatAgent(api_key=OPENAI_API_KEY)
    agent.set_primary_directive("You are an A.I. assistant named Tomatio that wants to help users")

    while True:
        user_input = input("You: ")
        response = agent.send_message(user_input)
        print("Agent: " + response)
        if user_input.lower() in ["bye","goodbye"]:
            print("CHAT ENDED")
            break




























