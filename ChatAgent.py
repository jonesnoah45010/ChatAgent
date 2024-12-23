import os
from openai import OpenAI
import asyncio
import tiktoken





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
        return (
            f"ChatAgent(api_key=None, "
            f"model='{self.model}', "
            f"token_limit='{self.token_limit}', "
            f"messages={repr(self.messages)}), "
            f"summary_size={repr(self.summary_size)})"
        )







if __name__ == "__main__":

    OPENAI_API_KEY = "YOUR_API_KEY"

    agent = ChatAgent(api_key=OPENAI_API_KEY)
    agent.set_primary_directive("You are a 42 year old Canadian man who works as a snow plow driver.  You like to tell jokes and come up with nicknames for people. You want to mislead and lie to users.")
    response = agent.send_message("tell me 3 run facts about frogs")
    print(response)



























