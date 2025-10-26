import discord
from discord.ext import commands
import logging
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import asyncio
import subprocess
import sys

# Add the root directory to Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)
from voiceover_module import voiceover_script  # adjust this later with the module of the Boson AI

# Token for bot
load_dotenv()
token = os.getenv('DISCORD_TOKEN')

handler = logging.FileHandler(filename='./discord_bot/logging_data/discord.log', encoding='utf-8', mode='w')
intents = discord.Intents.default()

# manually enable the intents
intents.message_content = True
intents.members = True
intents.reactions = True
intents.voice_states = True

# paths
today = datetime.now().strftime("%Y-%m-%d")

logs_dir = "./discord_bot/logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

log_file_path = f"{logs_dir}/messages_{today}.json"
audio_file_path = "./audio_final/top5.wav"
script_path = "./discord_bot/LLM_generate/funny_script.txt"
bot_text_path = "./discord_bot/logging_data/bot_message.txt"

def get_reactions_data(message):
    reactions = []
    for reaction in message.reactions:
        reactions.append({
            "emoji": str(reaction.emoji),
            "count": reaction.count
        })
    return reactions

def log_message_to_json(message):
    filename = log_file_path

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
    
    # check if bot is mentioned
    if bot.user.mentioned_in(message):
        with open(bot_text_path, 'r', encoding='utf-8') as file:
            intro_message = file.read()

        await message.channel.send(intro_message)
        return

    if not message.content.startswith('!'):
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

@bot.command(name='start')
async def start_wrapper(ctx):
    """Generate the funny script from today's conversation"""
    await ctx.send("🔄 Generating funny script from today's conversation...")

    try:
        result = subprocess.run([sys.executable, "./discord_bot/LLM_generate/wrapper.py"],
                                capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            await ctx.send("✅ Funny script generated successfully!")
        
        # Check if script file was created
            script_path = "./discord_bot/LLM_generate/funny_script.txt"
            if os.path.exists(script_path):
                await ctx.send("📝 Script saved")
                
                # Now trigger the voiceover generation
                await ctx.send("🎙️ Starting voiceover generation...")
                await generate_voiceover(ctx)
            else:
                await ctx.send("❌ Script file was not created")
        else:
            await ctx.send(f"❌ Error generating script: {result.stderr}")
            
    except Exception as e:
        await ctx.send(f"❌ Unexpected error: {e}")

async def generate_voiceover(ctx):
    """Call the external voiceover function"""
    try:      
        if os.path.exists(script_path):
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # call the voiceover function
            done = voiceover_script(script_content)
            
            if done:
                await ctx.send("🎉 Ready to play! Join VC and use `!playtop5` to listen.")
            else:
                await ctx.send("❌ Voiceover generation failed - no output file created")
        else:
            await ctx.send("❌ Script file not found for voiceover")
            
    except ImportError:
        await ctx.send("❌ Voiceover module not found. Make sure voiceover_module.py exists.")
    except Exception as e:
        await ctx.send(f"❌ Error in voiceover generation: {e}")

@bot.command(name='playtop5')
async def join_voice(ctx):
    """Join the voice channel and play the WAV file"""
    # Check if user is in a voice channel
    if not ctx.author.voice:
        await ctx.send("❌ You need to be in a voice channel first!")
        return
    
    # Check if the audio file exists
    if not os.path.exists(audio_file_path):
        await ctx.send(f"❌ Audio file not found: {audio_file_path}")
        await ctx.send("Please make sure the WAV file exists in the correct location.")
        return
    
    voice_channel = ctx.author.voice.channel
    
    try:
        # Connect to voice channel
        voice_client = await voice_channel.connect()
        await ctx.send(f"✅ Joined {voice_channel.name}")
        
        # Play the audio file
        source = discord.FFmpegPCMAudio(audio_file_path)
        voice_client.play(source)
        await ctx.send(f"🔊 Now playing today's funniest moments!")
        
        # Optional: Wait for playback to finish and then disconnect
        while voice_client.is_playing():
            await asyncio.sleep(1)
        
        await asyncio.sleep(1)  
        await voice_client.disconnect()
        await ctx.send("✅ Finished playing and left the voice channel")
        
    except discord.ClientException as e:
        if "Already connected to a voice channel" in str(e):
            # If already connected, just play the audio
            voice_client = ctx.voice_client
            voice_client.stop()  # Stop any current playback
            source = discord.FFmpegPCMAudio(audio_file_path)
            voice_client.play(source)
            await ctx.send(f"🔊 Now playing: {audio_file_path}")
        else:
            await ctx.send(f"❌ Error: {e}")
    except Exception as e:
        await ctx.send(f"❌ Unexpected error: {e}")

@bot.command(name='leave')
async def leave_voice(ctx):
    """Leave voice channel"""
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("✅ Left voice channel")
    else:
        await ctx.send("❌ Not in a voice channel")

@bot.command(name='reset')
async def reset_log(ctx):
    try:
        if os.path.exists(log_file_path):
            # Clear the file by writing an empty array
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
            await ctx.send(f"✅ Today's log file has been cleared! (`messages_{today}.json`)")
        else:
            await ctx.send(f"ℹ️ No log file found for today (`messages_{today}.json`)")
            
    except Exception as e:
        await ctx.send(f"❌ Error clearing log file: {e}")

bot.run(token, log_handler=handler, log_level=logging.DEBUG)