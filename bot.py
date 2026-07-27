import os

from dotenv import load_dotenv
from telegram import BotCommand, BotCommandScopeChat, MenuButtonCommands
from telegram.ext import Application

load_dotenv()

from auth import ADMIN_USER_IDS, auth_admin_commands, register_auth_handlers
from mess_control import message_control_commands, register_message_control_handlers
from symbol_control import register_symbol_handlers, symbol_control_commands

BOT_TOKEN = os.getenv("BOT_TOKEN")


def dedupe_commands(commands: list[BotCommand]) -> list[BotCommand]:
    seen: set[str] = set()
    result: list[BotCommand] = []
    for command in commands:
        if command.command in seen:
            continue
        seen.add(command.command)
        result.append(command)
    return result


async def setup_bot_menu(app: Application) -> None:
    # Menu chung cho user thường. Telegram không cho tạo command có tham số,
    # nên /stats BTC và /history BTC phải gõ tay, còn menu chỉ hiện /stats và /history.
    user_commands = dedupe_commands([
        BotCommand("start", "Bắt đầu"),
        BotCommand("whoami", "Lấy User ID"),
        *message_control_commands(),
        *symbol_control_commands(),
    ])

    await app.bot.set_my_commands(user_commands)

    # Menu riêng cho admin: có đủ lệnh user + lệnh quản trị để dễ bấm.
    admin_commands = dedupe_commands([
        *user_commands,
        *auth_admin_commands(),
    ])

    for admin_id in ADMIN_USER_IDS:
        await app.bot.set_my_commands(
            admin_commands,
            scope=BotCommandScopeChat(chat_id=admin_id),
        )
        print(f"[BOT_MENU] Registered {len(admin_commands)} admin commands for chat {admin_id}", flush=True)

    print(f"[BOT_MENU] Registered {len(user_commands)} default user commands", flush=True)

    await app.bot.set_chat_menu_button(menu_button=MenuButtonCommands())


def main() -> None:
    
    if not BOT_TOKEN:
        raise RuntimeError("Thiếu BOT_TOKEN trong file .env")

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(setup_bot_menu)
        .build()
    )

    register_auth_handlers(app)
    register_message_control_handlers(app)
    register_symbol_handlers(app)

    app.run_polling()


if __name__ == "__main__":
    main()
