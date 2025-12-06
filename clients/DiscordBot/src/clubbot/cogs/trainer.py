from __future__ import annotations

import discord
from discord import app_commands
from platform_core.errors import AppError, ErrorCode
from platform_core.job_events import default_events_channel
from platform_core.logging import get_logger
from platform_core.model_trainer_client import HTTPModelTrainerClient, ModelTrainerAPIError
from platform_discord.embed_helpers import add_field, create_embed, set_footer
from platform_discord.protocols import InteractionProto, UserProto, wrap_interaction
from platform_discord.rate_limiter import RateLimiter

from ..cogs.base import BaseCog, _BotProto
from ..config import DiscordbotSettings
from ..services.jobs.trainer_notifier import TrainerEventSubscriber


class TrainerCog(BaseCog):
    def __init__(
        self,
        bot: _BotProto | None,
        config: DiscordbotSettings,
        *,
        autostart_subscriber: bool = True,
    ) -> None:
        super().__init__()
        self.bot = bot
        self.config = config
        rl = int(self.config["model_trainer"]["rate_limit"])
        window = int(self.config["model_trainer"]["rate_window_seconds"])
        self.rate_limiter = RateLimiter(rl, window)
        self._subscriber: TrainerEventSubscriber | None = None
        self._autostart_subscriber = autostart_subscriber
        if self._autostart_subscriber:
            b = self.bot
            if b is not None:
                redis_url = (self.config["redis"]["redis_url"] or "").strip()
                if redis_url:
                    channel = default_events_channel("trainer")
                    self._subscriber = TrainerEventSubscriber(
                        bot=b, redis_url=redis_url, events_channel=channel
                    )
                    self._subscriber.start()
                    get_logger(__name__).info(
                        "Trainer events subscriber started (channel=%s)",
                        channel,
                    )

    # Lifecycle helper for orchestrator
    def ensure_subscriber_started(self) -> None:
        sub = self._subscriber
        if sub is not None:
            sub.start()

    def _mk_client(self) -> HTTPModelTrainerClient:
        base = (self.config["model_trainer"]["api_url"] or "").strip()
        if not base:
            raise AppError(
                ErrorCode.INVALID_INPUT, "Model Trainer API is not configured", http_status=400
            )
        return HTTPModelTrainerClient(
            base_url=base,
            api_key=(self.config["model_trainer"]["api_key"] or None),
            timeout_seconds=int(self.config["model_trainer"]["api_timeout_seconds"]),
        )

    @app_commands.command(name="train_model", description="Start a model training run")
    @app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
    @app_commands.allowed_installs(guilds=True, users=True)
    @app_commands.describe(
        model_family="Model family (e.g., gpt2)",
        model_size="Model size label (e.g., small)",
        max_seq_len="Max sequence length",
        num_epochs="Number of epochs",
        batch_size="Batch size",
        learning_rate="Learning rate",
        corpus_path="Path to corpus in API container (e.g., /data/corpus)",
        tokenizer_id="Tokenizer artifact ID",
    )
    async def train_model(
        self,
        interaction: discord.Interaction,
        model_family: str,
        model_size: str,
        max_seq_len: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        corpus_path: str,
        tokenizer_id: str,
    ) -> None:
        wrapped: InteractionProto = wrap_interaction(interaction)
        await self._train_model_impl(
            wrapped,
            model_family=model_family,
            model_size=model_size,
            max_seq_len=max_seq_len,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            corpus_path=corpus_path,
            tokenizer_id=tokenizer_id,
        )

    async def _train_model_impl(
        self,
        wrapped: InteractionProto,
        *,
        model_family: str,
        model_size: str,
        max_seq_len: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        corpus_path: str,
        tokenizer_id: str,
    ) -> None:
        """Internal implementation of train_model that accepts protocol types.

        This method contains the business logic and is designed for testability.
        """
        if not await self._safe_defer(wrapped, ephemeral=True):
            return
        request_id = self.new_request_id()
        log = self.request_logger(request_id)
        # InteractionProto guarantees a UserProto; avoid dynamic getattr best-effort.
        user_obj: UserProto = wrapped.user
        user_id = self.decode_int_attr(user_obj, "id")
        if user_id is None or int(user_id) <= 0:
            await self.handle_user_error(wrapped, log, "Could not determine your user id")
            return
        if not await self.check_rate_limit(
            wrapped,
            rate_limiter=self.rate_limiter,
            user_id=user_id,
            command="train_model",
            log=log,
            public_responses=False,
        ):
            return
        try:
            client = self._mk_client()
            res = await client.train(
                user_id=user_id,
                model_family=model_family,
                model_size=model_size,
                max_seq_len=max_seq_len,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                corpus_path=corpus_path,
                tokenizer_id=tokenizer_id,
                request_id=request_id,
            )
            await client.aclose()
        except AppError as e:
            log.info("User input error: %s", str(e))
            await self.handle_user_error(wrapped, log, str(e))
            return
        except ModelTrainerAPIError as e:
            log.info("Model trainer API error: %s", str(e))
            await self.handle_user_error(wrapped, log, f"API error: {e}")
            return
        # Unexpected exceptions: handle gracefully for users
        except (RuntimeError, ValueError, OSError, TypeError) as e:
            log.exception("Unhandled trainer error: %s", str(e))
            await self.handle_user_error(wrapped, log, "An error occurred. Please try again later.")
            return
        # Success path below
        embed = create_embed(
            title="Training Job Queued",
            description=(
                "Your training job has been queued successfully!\n"
                "You'll receive DM updates with progress and results."
            ),
            color=0x5865F2,
        )
        add_field(embed, name="Run ID", value=f"`{res['run_id']}`", inline=True)
        add_field(embed, name="Job ID", value=f"`{res['job_id']}`", inline=True)
        set_footer(embed, text=f"Request ID: {request_id}")
        await wrapped.followup.send(embed=embed, ephemeral=True)
        log.info("Queued training req=%s run=%s job=%s", request_id, res["run_id"], res["job_id"])
