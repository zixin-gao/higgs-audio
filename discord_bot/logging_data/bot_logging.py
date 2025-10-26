import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# Token for bot
load_dotenv()
token = os.getenv('DISCORD_TOKEN')

handler = logging.FileHandler(filename='./higgs-audio/discord_bot/logging_data/discord.log', encoding='utf-8', mode='w')
intents = discord.Intents.default()

# manually enable the intents
intents.message_content = True
intents.members = True
intents.reactions = True
intents.voice_states = True

# create logging dir
logs_dir = "./higgs-audio/discord_bot/logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

def get_reactions_data(message):
    reactions = []
    for reaction in message.reactions:
        reactions.append({
            "emoji": str(reaction.emoji),
            "count": reaction.count
        })
    return reactions

def log_message_to_json(message):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"{logs_dir}/messages_{today}.json"

    reactions_data = get_reactions_data(message)

    message_data = {
        "sender_name": str(message.author),
        "message_content": message.content,
        "message_id": str(message.id),
        "reactions": reactions_data
    }

    # if the file already exists, we append
    existing_data = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, Exception):
            existing_data = []

    # Check if message already exists in log
    message_found = False
    for i, existing_message in enumerate(existing_data):
        if existing_message.get("message_id") == str(message.id):
            # Replace the existing message with updated data
            existing_data[i] = message_data
            message_found = True
            break
    
    # If message not found, append it
    if not message_found:
        existing_data.append(message_data)

    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

# what the bot does
bot = commands.Bot(command_prefix = '!', intents=intents)

@bot.event
async def on_ready():
    print(f"We are ready to go in, {bot.user.name}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # log the message
    log_message_to_json(message)
    await bot.process_commands(message)

@bot.event
async def on_reaction_add(reaction, user):
    if user == bot.user:
        return
    
    message = reaction.message
    log_message_to_json(message)

@bot.event
async def on_reaction_remove(reaction, user):
    if user == bot.user:
        return
    
    message = reaction.message
    log_message_to_json(message)

bot.run(token, log_handler=handler, log_level=logging.DEBUG)