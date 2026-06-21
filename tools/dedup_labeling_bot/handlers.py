from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from research.dedup.annotation import (
    LABEL_DIFFERENT_PRODUCT,
    LABEL_EXACT_DUPLICATE,
    LABEL_UNCERTAIN,
)

from .callbacks import (
    DISCUSSION_LABEL_CALLBACK_PREFIX,
    NAV_CALLBACK_PREFIX,
    NEXT_CALLBACK,
    NOOP_CALLBACK_PREFIX,
    make_discussion_label_callback,
    make_label_callback,
    make_nav_callback,
    make_noop_callback,
    parse_label_callback,
    parse_nav_callback,
)
from .formatter import format_help_text, h
from .service import AssignedPair, LabelingBotService


def _label_button(row_index: int, label: str, text: str, selected_label: str | None) -> InlineKeyboardButton:
    button_text = f"✓ {text}" if selected_label == label else text
    callback_data = make_noop_callback(row_index) if selected_label == label else make_label_callback(row_index, label)
    return InlineKeyboardButton(button_text, callback_data=callback_data)


def build_keyboard(row_index: int, selected_label: str | None = None) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                _label_button(row_index, LABEL_EXACT_DUPLICATE, "Дубль", selected_label),
                _label_button(row_index, LABEL_DIFFERENT_PRODUCT, "Разные", selected_label),
            ],
            [
                _label_button(row_index, LABEL_UNCERTAIN, "Не уверен", selected_label),
            ],
            [
                InlineKeyboardButton("Назад", callback_data=make_nav_callback(row_index, "prev")),
                InlineKeyboardButton("Дальше", callback_data=make_nav_callback(row_index, "next")),
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
        BotCommand("team", "вступить в команду"),
        BotCommand("teams", "топ команд"),
        BotCommand("players", "топ игроков"),
        BotCommand("achievements", "мои ачивки"),
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
        self.combo_tasks: dict[int, asyncio.Task[None]] = {}

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
        await message.reply_text(
            "Готово, доступ открыт. Вступите в команду через /team <название>, "
            "потом нажмите /next, чтобы получить пару."
        )

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
        await message.reply_text(
            pair.message,
            reply_markup=build_keyboard(pair.row_index, pair.selected_label),
            parse_mode=ParseMode.HTML,
        )

    async def me(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_authorized(update):
            return
        message = update.effective_message
        if message is not None and update.effective_user is not None:
            async with self.lock:
                text = self.service.user_stats(update.effective_user.id)
            await message.reply_text(text)

    async def team(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_authorized(update):
            return
        user = update.effective_user
        message = update.effective_message
        if user is None or message is None:
            return
        if not context.args:
            await message.reply_text("Напишите название команды так: /team Команда мечты")
            return
        team_name = " ".join(context.args)
        async with self.lock:
            try:
                text = self.service.join_team(user.id, team_name)
            except ValueError as exc:
                await message.reply_text(f"Не получилось вступить в команду: {exc}")
                return
        await message.reply_text(text, parse_mode=ParseMode.HTML)
        await self._refresh_leaderboard_pin(context)

    async def teams(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_authorized(update):
            return
        message = update.effective_message
        if message is None:
            return
        async with self.lock:
            text = self.service.team_leaderboard()
        await message.reply_text(text, parse_mode=ParseMode.HTML)

    async def players(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_authorized(update):
            return
        message = update.effective_message
        if message is None:
            return
        async with self.lock:
            text = self.service.player_leaderboard()
        await message.reply_text(text, parse_mode=ParseMode.HTML)

    async def achievements(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._ensure_authorized(update):
            return
        user = update.effective_user
        message = update.effective_message
        if user is None or message is None:
            return
        async with self.lock:
            text = self.service.user_achievements(user.id)
        await message.reply_text(text, parse_mode=ParseMode.HTML)

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
        if not self.service.is_authorized(user.id):
            await query.answer()
            await self._safe_edit_message(query, "Сначала войдите через /start <пароль>.")
            return

        data = query.data or ""
        if data.startswith(f"{DISCUSSION_LABEL_CALLBACK_PREFIX}:"):
            await self._handle_discussion_callback(query, context, user, data)
            return

        if data.startswith(f"{NAV_CALLBACK_PREFIX}:"):
            await self._handle_navigation_callback(query, user, data)
            return

        if data.startswith(f"{NOOP_CALLBACK_PREFIX}:"):
            await query.answer("Уже сохранено.")
            return

        if data == NEXT_CALLBACK:
            await query.answer()
            pair = await self._next_pair(user.id)
            if pair is None:
                await self._safe_edit_message(query, "Свободных пар больше нет.")
                return
            await self._edit_pair_message(query, pair)
            return

        try:
            parsed = parse_label_callback(data)
        except Exception:
            await query.answer()
            await self._safe_edit_message(query, "Не понял кнопку. Нажмите /next для новой пары.")
            return

        if parsed.label == LABEL_UNCERTAIN and self.service.config.discussion_chat_id is not None:
            await self._handle_uncertain_callback(query, context, user, parsed.row_index)
            return

        async with self.lock:
            try:
                outcome = self.service.label_row(user.id, parsed.row_index, parsed.label)
            except ValueError as exc:
                next_pair = self.service.navigate_pair(user.id, parsed.row_index, "next")
                await query.answer()
                if next_pair is None:
                    await self._safe_edit_message(query, f"{exc}. Свободных пар больше нет.")
                    return
                await self._safe_edit_message(
                    query,
                    next_pair.message,
                    reply_markup=build_keyboard(next_pair.row_index, next_pair.selected_label),
                    parse_mode=ParseMode.HTML,
                )
                return
            next_pair = self.service.navigate_pair(user.id, parsed.row_index, "next")
            current_pair = self.service.pair_for_row(user.id, parsed.row_index)

        await query.answer(self._toast(outcome.message))
        if next_pair is None:
            await self._edit_pair_message(query, current_pair)
            await self._handle_outcome_notifications(context, outcome)
            return
        await self._edit_pair_message(query, next_pair)
        await self._handle_outcome_notifications(context, outcome)

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

    @staticmethod
    def _toast(text: str) -> str:
        return text if len(text) <= 180 else f"{text[:177]}..."

    async def _safe_edit_message(
        self,
        query: object,
        text: str,
        *,
        reply_markup: InlineKeyboardMarkup | None = None,
        parse_mode: str | None = None,
    ) -> None:
        try:
            await query.edit_message_text(
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode,
            )
        except BadRequest as exc:
            if "Message is not modified" in str(exc):
                return
            raise

    async def _edit_pair_message(self, query: object, pair: AssignedPair) -> None:
        await self._safe_edit_message(
            query,
            pair.message,
            reply_markup=build_keyboard(pair.row_index, pair.selected_label),
            parse_mode=ParseMode.HTML,
        )

    async def _handle_navigation_callback(self, query: object, user: object, data: str) -> None:
        try:
            parsed = parse_nav_callback(data)
        except Exception:
            await query.answer()
            await self._safe_edit_message(query, "Не понял навигацию. Нажмите /next для новой пары.")
            return

        async with self.lock:
            pair = self.service.navigate_pair(user.id, parsed.row_index, parsed.direction)

        if pair is None:
            text = "Это первая пара." if parsed.direction == "prev" else "Дальше пока нет."
            await query.answer(text)
            return
        await query.answer()
        await self._edit_pair_message(query, pair)

    async def _broadcast_milestones(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        milestones: object,
    ) -> None:
        if not milestones:
            return
        recipients: list[int | str] = list(self.service.milestone_recipients())
        discussion_chat_id = self.service.config.discussion_chat_id
        if discussion_chat_id is not None:
            recipients.append(discussion_chat_id)
        targets = list(dict.fromkeys(recipients))

        async def send_one(chat_id: int | str, text: str) -> None:
            try:
                await context.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
            except Exception:
                return

        for milestone in milestones:
            await asyncio.gather(*(send_one(chat_id, milestone.message) for chat_id in targets))

    async def _handle_outcome_notifications(self, context: ContextTypes.DEFAULT_TYPE, outcome: object) -> None:
        await self._broadcast_milestones(context, getattr(outcome, "milestones", ()))
        self._schedule_combo_timers(context, getattr(outcome, "combo_timers", ()))
        await self._broadcast_game_announcements(context, getattr(outcome, "game_announcements", ()))
        if getattr(outcome, "combo_timers", ()) or getattr(outcome, "game_announcements", ()):
            await self._refresh_leaderboard_pin(context)

    async def _broadcast_game_announcements(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        announcements: object,
    ) -> None:
        if not announcements:
            return

        async def send_one(chat_id: int | str, text: str) -> None:
            try:
                await context.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
            except Exception:
                return

        for announcement in announcements:
            if getattr(announcement, "group_only", False):
                targets = self.service.group_announcement_recipients()
            else:
                targets = self.service.announcement_recipients()
            if not targets:
                continue
            await asyncio.gather(*(send_one(chat_id, announcement.message) for chat_id in targets))

    def _schedule_combo_timers(self, context: ContextTypes.DEFAULT_TYPE, timers: object) -> None:
        for timer in timers or ():
            existing_task = self.combo_tasks.pop(timer.team_id, None)
            if existing_task is not None:
                existing_task.cancel()
            delay = max(0.0, (timer.deadline_at - datetime.now(timezone.utc)).total_seconds())
            task = asyncio.create_task(self._combo_timeout_task(context, timer.team_id, timer.deadline_at, delay))
            self.combo_tasks[timer.team_id] = task
            task.add_done_callback(lambda _task, team_id=timer.team_id: self.combo_tasks.pop(team_id, None))

    async def _combo_timeout_task(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        team_id: int,
        deadline_at: datetime,
        delay: float,
    ) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

        async with self.lock:
            announcement = self.service.expire_team_combo(team_id=team_id, deadline_at=deadline_at)
        if announcement is None:
            return
        await self._broadcast_game_announcements(context, (announcement,))
        await self._refresh_leaderboard_pin(context)

    async def _refresh_leaderboard_pin(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = self.service.config.leaderboard_chat_id
        if chat_id is None:
            return
        async with self.lock:
            text = self.service.team_leaderboard()
            pin = self.service.leaderboard_pin()

        async def send_new_pin() -> None:
            message = await context.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
            try:
                await context.bot.pin_chat_message(
                    chat_id=chat_id,
                    message_id=message.message_id,
                    disable_notification=True,
                )
            except Exception:
                pass
            async with self.lock:
                self.service.record_leaderboard_pin(chat_id=chat_id, message_id=message.message_id)

        if pin is not None and str(pin[0]) == str(chat_id):
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=pin[1],
                    text=text,
                    parse_mode=ParseMode.HTML,
                )
                return
            except BadRequest as exc:
                if "Message is not modified" in str(exc):
                    return
            except Exception:
                return

        try:
            await send_new_pin()
        except Exception:
            return

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

    async def _handle_discussion_callback(
        self,
        query: object,
        context: ContextTypes.DEFAULT_TYPE,
        user: object,
        data: str,
    ) -> None:
        if not self.service.is_authorized(user.id):
            await query.answer()
            await self._safe_edit_message(query, "Сначала войдите через /start <пароль>.")
            return
        try:
            parsed = parse_label_callback(data)
        except Exception:
            await query.answer()
            await self._safe_edit_message(query, "Не понял кнопку обсуждения.")
            return
        async with self.lock:
            outcome = self.service.vote_discussion_row(user.id, parsed.row_index, parsed.label)
            text = self.service.discussion_message(parsed.row_index, self._user_display(user))
        await query.answer(self._toast(outcome.message))
        if outcome.final_label is not None:
            text = f"{text}\n\n<b>Итог:</b> {outcome.final_label}"
            reply_markup = None
        else:
            text = f"{text}\n\n<b>Статус:</b> голос сохранён, ждём консенсус."
            reply_markup = build_discussion_keyboard(parsed.row_index)
        await self._safe_edit_message(
            query,
            text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
        )
        await self._handle_outcome_notifications(context, outcome)

    async def _handle_uncertain_callback(
        self,
        query: object,
        context: ContextTypes.DEFAULT_TYPE,
        user: object,
        row_index: int,
    ) -> None:
        await query.answer("Отправляю в обсуждение.")
        chat_id = self.service.config.discussion_chat_id
        async with self.lock:
            should_send = self.service.should_send_discussion(row_index)
            discussion_message = (
                self.service.discussion_message(row_index, self._user_display(user))
                if should_send
                else None
            )
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
                    await self._safe_edit_message(query, text)
                    return
                await self._safe_edit_message(
                    query,
                    f"{h(text)}\n\n{current_pair.message}",
                    reply_markup=build_keyboard(current_pair.row_index, current_pair.selected_label),
                    parse_mode=ParseMode.HTML,
                )
                return
            message_id = int(send_result)

        async with self.lock:
            try:
                self.service.move_row_to_discussion(user.id, row_index, LABEL_UNCERTAIN)
            except ValueError as exc:
                next_pair = self.service.navigate_pair(user.id, row_index, "next")
                if next_pair is None:
                    await self._safe_edit_message(query, f"{exc}. Свободных пар больше нет.")
                    return
                await self._safe_edit_message(
                    query,
                    next_pair.message,
                    reply_markup=build_keyboard(next_pair.row_index, next_pair.selected_label),
                    parse_mode=ParseMode.HTML,
                )
                return
            if message_id is not None and chat_id is not None:
                self.service.record_discussion_post(
                    row_index,
                    user.id,
                    chat_id=chat_id,
                    message_id=message_id,
                )
            next_pair = self.service.navigate_pair(user.id, row_index, "next")
            current_pair = self.service.pair_for_row(user.id, row_index)

        if next_pair is None:
            await self._edit_pair_message(query, current_pair)
            return
        await self._edit_pair_message(query, next_pair)


def create_application(service: LabelingBotService) -> Application:
    handlers = TelegramLabelingHandlers(service)
    application = Application.builder().token(service.config.token).post_init(set_bot_commands).build()
    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(CommandHandler("help", handlers.help))
    application.add_handler(CommandHandler("menu", handlers.help))
    application.add_handler(CommandHandler("next", handlers.next))
    application.add_handler(CommandHandler("me", handlers.me))
    application.add_handler(CommandHandler("team", handlers.team))
    application.add_handler(CommandHandler("teams", handlers.teams))
    application.add_handler(CommandHandler("players", handlers.players))
    application.add_handler(CommandHandler("achievements", handlers.achievements))
    application.add_handler(CommandHandler("stats", handlers.stats))
    application.add_handler(CommandHandler("release", handlers.release))
    application.add_handler(CommandHandler("logout", handlers.logout))
    application.add_handler(CallbackQueryHandler(handlers.callback))
    return application
