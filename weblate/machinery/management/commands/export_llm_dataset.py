# Copyright © Michal Čihař <michal@weblate.org>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Export existing translations as an LLM fine-tuning dataset."""

from __future__ import annotations

import json
from itertools import chain
from typing import TYPE_CHECKING

from django.core.management.base import CommandError

from weblate.glossary.models import (
    fetch_glossary_terms,
    get_glossary_terms,
    get_glossary_tuples,
)
from weblate.machinery.llm import PROMPT, BaseLLMTranslation
from weblate.utils.management.base import WeblateLangCommand
from weblate.utils.state import STATE_TRANSLATED

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from django.core.management.base import CommandParser

    from weblate.trans.models import Translation
    from weblate.trans.models.unit import Unit


def _format_prompt_part(text: str) -> str:
    text = text.strip()
    if text and not text.endswith("."):
        text = f"{text}."
    return text


def _build_system_prompt(persona: str, style: str) -> str:
    return PROMPT.format(
        persona=_format_prompt_part(persona),
        style=_format_prompt_part(style),
    )


def _batched(iterable: Iterable[Unit], size: int) -> Iterator[list[Unit]]:
    batch: list[Unit] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class Command(WeblateLangCommand):
    """Export existing translations as LLM fine-tuning dataset."""

    help = (
        "exports existing translations as an LLM fine-tuning dataset "
        "(JSONL chat format, compatible with datasets.load_dataset)"
    )

    written = 0
    skipped = 0

    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--output",
            required=True,
            help="Path to the JSONL output file",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help=(
                "Number of units bundled into a single training example "
                "(default: 1)"
            ),
        )
        parser.add_argument(
            "--persona",
            default="",
            help="Optional persona text to inject into the system prompt",
        )
        parser.add_argument(
            "--style",
            default="",
            help="Optional style text to inject into the system prompt",
        )
        parser.add_argument(
            "--skip-system",
            action="store_true",
            help="Skip injecting the system prompt",
        )

    def _iter_training_units(self, translation: Translation) -> Iterator[Unit]:
        units = (
            translation.unit_set.filter(state__gte=STATE_TRANSLATED)
            .exclude(target="")
            .exclude(check__name="same")
            .order_by("pk")
            .prefetch()
        )
        for unit in units:
            if unit.readonly:
                continue
            target_plurals = unit.get_target_plurals()
            if not target_plurals or not all(target_plurals):
                continue
            yield unit

    def _build_example(
        self,
        prompt: str | None,
        batch: list[Unit],
        source_code: str,
        target_code: str,
    ) -> dict | None:
        fetch_glossary_terms(batch)
        glossary = dict(
            get_glossary_tuples(
                chain.from_iterable(
                    get_glossary_terms(unit, include_variants=False)
                    for unit in batch
                )
            )
        )

        strings: list[dict[str, str]] = []
        responses: list[str] = []
        for unit in batch:
            source_raw = unit.get_source_plurals()[0]
            source_text, _specs = BaseLLMTranslation._cleanup_source_variant(
                source_raw, unit
            )
            target_raw = unit.get_target_plurals()[0]
            target_text = BaseLLMTranslation._placeholderize_translation(
                target_raw, source_text, unit
            )
            if target_text is None:
                self.skipped += 1
                continue
            strings.append({"source": source_text})
            responses.append(target_text)

        if len(strings) == 0:
            return None
        user_content = json.dumps(
            {
                "source_language": source_code,
                "target_language": target_code,
                "glossary": glossary,
                "strings": strings,
            },
            ensure_ascii=False,
        )
        assistant_content = json.dumps(responses, ensure_ascii=False)
        messages: list[dict[str, str]] = []
        if prompt is not None:
            messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
        return {"messages": messages}

    def handle(self, *args, **options) -> None:
        batch_size = options["batch_size"]
        if batch_size < 1:
            msg = "--batch-size must be at least 1"
            raise CommandError(msg)

        translations = self.get_translations(**options).exclude_source()
        prompt: str | None = (
            None
            if options["skip_system"]
            else _build_system_prompt(options["persona"], options["style"])
        )


        with open(options["output"], "w", encoding="utf-8") as fh:
            for translation in translations:
                source_code = translation.component.source_language.code
                target_code = translation.language.code
                if source_code == target_code:
                    continue
                for batch in _batched(
                    self._iter_training_units(translation), batch_size
                ):
                    example = self._build_example(
                        prompt, batch, source_code, target_code
                    )
                    if example is None:
                        continue
                    fh.write(json.dumps(example, ensure_ascii=False))
                    fh.write("\n")
                    self.written += 1

        self.stdout.write(
            f"Exported {self.written} example(s); "
            f"skipped {self.skipped} unit(s) due to placeholder mismatches."
        )
