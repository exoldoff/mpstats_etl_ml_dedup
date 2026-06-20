from __future__ import annotations

import asyncio

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from research.dedup.annotation import (
    LABEL_DIFFERENT_PRODUCT,
    LABEL_EXACT_DUPLICATE,
    LABEL_UNCERTAIN,
)

from .callbacks import NEXT_CALLBACK, make_label_callback, parse_label_callback
from .formatter import format_help_text
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
            await message.reply_text(format_help_text())

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
        await message.reply_text(pair.message, reply_markup=build_keyboard(pair.row_index))

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
        if data == NEXT_CALLBACK:
            pair = await self._next_pair(user.id)
            if pair is None:
                await query.edit_message_text("Свободных пар больше нет.")
                return
            await query.edit_message_text(pair.message, reply_markup=build_keyboard(pair.row_index))
            return

        try:
            parsed = parse_label_callback(data)
        except Exception:
            await query.edit_message_text("Не понял кнопку. Нажмите /next для новой пары.")
            return

        async with self.lock:
            try:
                outcome = self.service.label_row(user.id, parsed.row_index, parsed.label)
            except ValueError as exc:
                await query.edit_message_text(f"{exc}. Нажмите /next, чтобы получить актуальную пару.")
                return
            next_pair = self.service.next_pair(user.id)

        if next_pair is None:
            await query.edit_message_text(f"{outcome.message}\n\nСвободных пар больше нет.")
            return
        await query.edit_message_text(
            f"{outcome.message}\n\n{next_pair.message}",
            reply_markup=build_keyboard(next_pair.row_index),
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


def create_application(service: LabelingBotService) -> Application:
    handlers = TelegramLabelingHandlers(service)
    application = Application.builder().token(service.config.token).build()
    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(CommandHandler("help", handlers.help))
    application.add_handler(CommandHandler("next", handlers.next))
    application.add_handler(CommandHandler("me", handlers.me))
    application.add_handler(CommandHandler("stats", handlers.stats))
    application.add_handler(CommandHandler("release", handlers.release))
    application.add_handler(CommandHandler("logout", handlers.logout))
    application.add_handler(CallbackQueryHandler(handlers.callback))
    return application
