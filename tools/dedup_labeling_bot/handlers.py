from __future__ import annotations

import asyncio

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from research.dedup.annotation import (
    LABEL_DIFFERENT_PRODUCT,
    LABEL_EXACT_DUPLICATE,
    LABEL_UNCERTAIN,
)

from .callbacks import (
    DISCUSSION_LABEL_CALLBACK_PREFIX,
    NEXT_CALLBACK,
    make_discussion_label_callback,
    make_label_callback,
    parse_label_callback,
)
from .formatter import format_help_text, h
from .service import AssignedPair, LabelingBotService


def build_keyboard(row_index: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Дубль", callback_data=make_label_callback(row_index, LABEL_EXACT_DUPLICATE)),
                InlineKeyboardButton("Разные", callback_data=make_label_callback(row_index, LABEL_DIFFERENT_PRODUCT)),
            ],
            [
                InlineKeyboardButton("Не уверен", callback_data=make_label_callback(row_index, LABEL_UNCERTAIN)),
                InlineKeyboardButton("Дальше", callback_data=NEXT_CALLBACK),
            ],
        ]
    )


def build_discussion_keyboard(row_index: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Дубль",
                    callback_data=make_discussion_label_callback(row_index, LABEL_EXACT_DUPLICATE),
                ),
                InlineKeyboardButton(
                    "Разные",
                    callback_data=make_discussion_label_callback(row_index, LABEL_DIFFERENT_PRODUCT),
                ),
            ],
            [
                InlineKeyboardButton(
                    "Не уверен",
                    callback_data=make_discussion_label_callback(row_index, LABEL_UNCERTAIN),
                ),
            ],
        ]
    )


def build_bot_commands() -> list[BotCommand]:
    return [
        BotCommand("start", "войти по паролю"),
        BotCommand("next", "получить пару"),
        BotCommand("me", "моя статистика"),
        BotCommand("stats", "общий прогресс"),
        BotCommand("release", "освободить мои пары"),
        BotCommand("logout", "выйти"),
        BotCommand("menu", "показать команды"),
        BotCommand("help", "помощь"),
    ]


async def set_bot_commands(application: Application) -> None:
    await application.bot.set_my_commands(build_bot_commands())


class TelegramLabelingHandlers:
    def __init__(self, service: LabelingBotService) -> None:
        self.service = service
        self.lock = asyncio.Lock()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        message = update.effective_message
        if user is None or message is None:
            return
        if not context.args:
            await message.reply_text("Введите пароль так: /start <пароль>")
            return
        password = " ".join(context.args)
        async with self.lock:
            ok = self.service.authorize(
                user.id,
                password,
                username=user.username or "",
                first_name=user.first_name or "",
            )
        if not ok:
            await message.reply_text("Пароль не подошёл.")
            return
        await message.reply_text("Готово, доступ открыт. Нажмите /next, чтобы получить пару.")

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_authorized(update):
            return
        message = update.effective_message
        if message is not None:
            await message.reply_text(format_help_text(), parse_mode=ParseMode.HTML)

    async def next(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_authorized(update):
            return
        message = update.effective_message
        if message is None or update.effective_user is None:
            return
        pair = await self._next_pair(update.effective_user.id)
        if pair is None:
            await message.reply_text("Свободных пар больше нет.")
            return
        await message.reply_text(pair.message, reply_markup=build_keyboard(pair.row_index), parse_mode=ParseMode.HTML)

    async def me(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_authorized(update):
            return
        message = update.effective_message
        if message is not None and update.effective_user is not None:
            async with self.lock:
                text = self.service.user_stats(update.effective_user.id)
            await message.reply_text(text)

    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_authorized(update):
            return
        user = update.effective_user
        message = update.effective_message
        if user is None or message is None:
            return
        admins = self.service.config.admin_user_ids
        if admins and user.id not in admins:
            await message.reply_text("Общая статистика доступна только админам. Ваша статистика: /me")
            return
        async with self.lock:
            text = self.service.stats()
        await message.reply_text(text)

    async def release(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_authorized(update):
            return
        user = update.effective_user
        message = update.effective_message
        if user is None or message is None:
            return
        async with self.lock:
            released = self.service.release_user_assignments(user.id)
        await message.reply_text(f"Освобождено активных пар: {released}")

    async def logout(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_authorized(update):
            return
        user = update.effective_user
        message = update.effective_message
        if user is None or message is None:
            return
        async with self.lock:
            released = self.service.logout(user.id)
        await message.reply_text(f"Вы вышли. Освобождено активных пар: {released}")

    async def callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        user = update.effective_user
        if query is None or user is None:
            return
        await query.answer()
        if not self.service.is_authorized(user.id):
            await query.edit_message_text("Сначала войдите через /start <пароль>.")
            return

        data = query.data or ""
        if data.startswith(f"{DISCUSSION_LABEL_CALLBACK_PREFIX}:"):
            await self._handle_discussion_callback(query, user, data)
            return

        if data == NEXT_CALLBACK:
            pair = await self._next_pair(user.id)
            if pair is None:
                await query.edit_message_text("Свободных пар больше нет.")
                return
            await query.edit_message_text(
                pair.message,
                reply_markup=build_keyboard(pair.row_index),
                parse_mode=ParseMode.HTML,
            )
            return

        try:
            parsed = parse_label_callback(data)
        except Exception:
            await query.edit_message_text("Не понял кнопку. Нажмите /next для новой пары.")
            return

        if parsed.label == LABEL_UNCERTAIN and self.service.config.discussion_chat_id is not None:
            await self._handle_uncertain_callback(query, context, user, parsed.row_index)
            return

        async with self.lock:
            try:
                outcome = self.service.label_row(user.id, parsed.row_index, parsed.label)
            except ValueError as exc:
                await query.edit_message_text(f"{exc}. Нажмите /next, чтобы получить актуальную пару.")
                return
            next_pair = self.service.next_pair(user.id)

        response_text = outcome.message

        if next_pair is None:
            await query.edit_message_text(f"{response_text}\n\nСвободных пар больше нет.")
            return
        await query.edit_message_text(
            f"{response_text}\n\n{next_pair.message}",
            reply_markup=build_keyboard(next_pair.row_index),
            parse_mode=ParseMode.HTML,
        )

    async def _ensure_authorized(self, update: Update) -> bool:
        user = update.effective_user
        message = update.effective_message
        if user is None:
            return False
        if not self.service.is_authorized(user.id):
            if message is not None:
                await message.reply_text("Сначала войдите через /start <пароль>.")
            return False
        self.service.touch_user(user.id, username=user.username or "", first_name=user.first_name or "")
        return True

    async def _next_pair(self, user_id: int) -> AssignedPair | None:
        async with self.lock:
            return self.service.next_pair(user_id)

    async def _send_discussion_message(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        row_index: int,
        user_id: int,
        chat_id: int | str | None,
        text: str | None,
    ) -> str | None:
        if text is None:
            return None
        if chat_id is None:
            return "Общий чат не настроен: задайте DEDUP_TELEGRAM_DISCUSSION_CHAT_ID."
        try:
            message = await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=build_discussion_keyboard(row_index),
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:
            return f"Не смог отправить в общий чат: {exc}"
        return str(message.message_id)

    @staticmethod
    def _user_display(user: object) -> str:
        user_id = getattr(user, "id", "")
        username = getattr(user, "username", "") or ""
        first_name = getattr(user, "first_name", "") or ""
        if username:
            return f"@{username} ({user_id})"
        if first_name:
            return f"{first_name} ({user_id})"
        return str(user_id)

    async def _handle_discussion_callback(self, query: object, user: object, data: str) -> None:
        if not self.service.is_authorized(user.id):
            await query.edit_message_text("Сначала войдите через /start <пароль>.")
            return
        try:
            parsed = parse_label_callback(data)
        except Exception:
            await query.edit_message_text("Не понял кнопку обсуждения.")
            return
        async with self.lock:
            outcome = self.service.vote_discussion_row(user.id, parsed.row_index, parsed.label)
            text = self.service.discussion_message(parsed.row_index, self._user_display(user))
        if outcome.final_label is not None:
            text = f"{text}\n\n<b>Итог:</b> {outcome.final_label}"
            reply_markup = None
        else:
            text = f"{text}\n\n<b>Статус:</b> голос сохранён, ждём консенсус."
            reply_markup = build_discussion_keyboard(parsed.row_index)
        await query.edit_message_text(
            text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
        )

    async def _handle_uncertain_callback(
        self,
        query: object,
        context: ContextTypes.DEFAULT_TYPE,
        user: object,
        row_index: int,
    ) -> None:
        chat_id = self.service.config.discussion_chat_id
        async with self.lock:
            should_send = self.service.should_send_discussion(row_index)
            discussion_message = (
                self.service.discussion_message(row_index, self._user_display(user))
                if should_send
                else None
            )
        discussion_note = "Отправил пару в общий чат для обсуждения."
        message_id: int | None = None
        if discussion_message is not None:
            send_result = await self._send_discussion_message(
                context,
                row_index=row_index,
                user_id=user.id,
                chat_id=chat_id,
                text=discussion_message,
            )
            if send_result is None or not send_result.isdigit():
                async with self.lock:
                    current_pair = self.service.next_pair(user.id)
                text = send_result or "Не смог отправить в общий чат."
                if current_pair is None:
                    await query.edit_message_text(text)
                    return
                await query.edit_message_text(
                    f"{h(text)}\n\n{current_pair.message}",
                    reply_markup=build_keyboard(current_pair.row_index),
                    parse_mode=ParseMode.HTML,
                )
                return
            message_id = int(send_result)
        else:
            discussion_note = "Пара уже есть в общем чате для обсуждения."

        async with self.lock:
            try:
                outcome = self.service.move_row_to_discussion(user.id, row_index, LABEL_UNCERTAIN)
            except ValueError as exc:
                await query.edit_message_text(f"{exc}. Нажмите /next, чтобы получить актуальную пару.")
                return
            if message_id is not None and chat_id is not None:
                self.service.record_discussion_post(
                    row_index,
                    user.id,
                    chat_id=chat_id,
                    message_id=message_id,
                )
            next_pair = self.service.next_pair(user.id)

        response_text = f"{outcome.message}\n{discussion_note}"
        if next_pair is None:
            await query.edit_message_text(f"{response_text}\n\nСвободных пар больше нет.")
            return
        await query.edit_message_text(
            f"{response_text}\n\n{next_pair.message}",
            reply_markup=build_keyboard(next_pair.row_index),
            parse_mode=ParseMode.HTML,
        )


def create_application(service: LabelingBotService) -> Application:
    handlers = TelegramLabelingHandlers(service)
    application = Application.builder().token(service.config.token).post_init(set_bot_commands).build()
    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(CommandHandler("help", handlers.help))
    application.add_handler(CommandHandler("menu", handlers.help))
    application.add_handler(CommandHandler("next", handlers.next))
    application.add_handler(CommandHandler("me", handlers.me))
    application.add_handler(CommandHandler("stats", handlers.stats))
    application.add_handler(CommandHandler("release", handlers.release))
    application.add_handler(CommandHandler("logout", handlers.logout))
    application.add_handler(CallbackQueryHandler(handlers.callback))
    return application
