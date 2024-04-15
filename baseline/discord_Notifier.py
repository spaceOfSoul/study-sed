import discord
import os
from dotenv import load_dotenv

class DiscordNotifier:
    def __init__(self, token,user_id, message):
        self.token = token
        self.user_id = user_id
        intents = discord.Intents.default()
        intents.messages = True
        intents.dm_messages = True
        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_ready():
            print(f'Logged in as {self.client.user}')
            await self.send_message(self.user_id, message)
            await self.client.close()

    async def send_message(self, user_id, message):
        await self.client.wait_until_ready()
        user = await self.client.fetch_user(user_id)
        await user.send(message)

if __name__ == '__main__':
    load_dotenv()
    TOKEN = os.getenv('DISCORD_TOKEN')
    USER_ID = os.getenv('USER_ID')
    notifier = DiscordNotifier(TOKEN,USER_ID, 'test 2')
    notifier.client.run(TOKEN)
