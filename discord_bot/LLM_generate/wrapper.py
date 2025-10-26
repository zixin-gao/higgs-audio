import openai
import os
import json
from datetime import datetime


client = openai.Client(
    api_key=BOSON_API_KEY,
    base_url="https://hackathon.boson.ai/v1"
)

def get_today_conversation():
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = f"./higgs-audio/discord_bot/logs/messages_{today}.json"

    if not os.path.exists(log_file):
        return "No conversations today."

    with open(log_file, 'r', encoding='utf-8') as f:
        try:
            messages = json.load(f)
        except json.JSONDecodeError:
            return "Error reading log file."
        
        conversation_text = ""
        for msg in messages:
            conversation_text += f"{msg['sender_name']}: {msg['message_content']}\n"
            if msg['reactions']:
                conversation_text += f"Reactions: {', '.join([f'{r['emoji']}({r['count']})' for r in msg['reactions']])}\n"
            conversation_text += "\n"
    
    return conversation_text

conversation_text = get_today_conversation()
response = client.chat.completions.create(
    model="Qwen3-32B-thinking-Hackathon", 
    messages=[
        {"role": "system", "content": "You are a creative, witty, and humorous AI wrapper!"},
        {"role": "user", "content": f"We've given you a log of messages in a group of friends' Discord server here: {conversation_text} Don't print out your thinking. Only print out your script. Do not include emojis in the script. Choose the five funniest messages, and come up with a reason why each of them are funny. If a message has reactions, it is more likely to be funny, but don't rely on them exclusively. The discord name of whoever said the message should be logged as the speaker. Begin with a fun intro. It should be styled like a podcast or top five-style video. Format the final message like this: [Narrator] Welcome to today's funniest moments! This is a funny introduction. Number one. \n [USER1] funny message. \n[Narrator] explains why in a few funny and witty sentences. Do this for five funny messages. Have the narrator announce the number of each message before the message is said. At the end, come up a witty summary of the funniest messages for the narrator to say. Do not include your thinking as part of the response."}
    ],
    max_tokens=2048,
    temperature=0.7
)

print(response.choices[0].message.content)