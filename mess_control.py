from pathlib import Path

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes

HELP_FILE_PATH = Path("help.txt")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message

    if not message:
        return

    if not HELP_FILE_PATH.exists():
        await message.reply_text("Chưa có file help.txt.")
        return

    help_text = HELP_FILE_PATH.read_text(encoding="utf-8").strip()

    if not help_text:
        await message.reply_text("File help.txt đang trống.")
        return

    await message.reply_text(help_text)


def register_message_control_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("help", help_command))


def message_control_commands() -> list[BotCommand]:
    return [
        BotCommand("help", "Giúp đỡ"),
    ]
