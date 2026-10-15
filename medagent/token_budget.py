"""Preflight context budgets. Byte estimates are conservative, not native token counts."""
import json
import math


class ContextBudgetExceeded(ValueError):
    pass


DEFAULT_CONTEXT = {
    'context_window_tokens': 32768, 'output_reserve_tokens': 4096, 'safety_tokens': 512,
    'specialty_tokens': 10000, 'policy_tokens': 10000, 'summary_tokens': 1200,
    'active_facts': 64, 'active_sources': 8, 'active_messages': 8,
    'evidence_items': 48, 'evidence_window_chars': 800, 'archive_candidates': 64,
    'image_tokens_minimum': 4096, 'image_patch_pixels': 14, 'image_tokens_per_patch': 4,
}


def context_options(settings=None):
    if settings is None:
        from .settings import get_settings
        settings = get_settings()
    return {**DEFAULT_CONTEXT, **settings.data.get('context', {})}


class TokenCounter:
    """UTF-8 byte upper estimate for byte-based tokenizers, plus message framing."""
    method = 'utf8_bytes_conservative_estimate'

    def count(self, value):
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        return len(text.encode('utf-8'))


class TokenBudget:
    def __init__(self, options=None, model=None, counter=None):
        self.options = {**DEFAULT_CONTEXT, **(options or {})}
        self.model = model or {}
        self.counter = counter or TokenCounter()
        self.window = self.model.get('context_window_tokens', self.options['context_window_tokens'])
        self.safety = self.options['safety_tokens']
        self.output = self.model.get('max_tokens', self.options['output_reserve_tokens'])

    def input_limit(self):
        return self.window - self.output - self.safety

    def check(self, input_tokens, output_tokens=None, method=None):
        output = self.output if output_tokens is None else output_tokens
        if type(input_tokens) is not int or input_tokens < 0 or input_tokens + output + self.safety > self.window:
            raise ContextBudgetExceeded('context_budget_exceeded:input=' + str(input_tokens) +
                ',output=' + str(output) + ',window=' + str(self.window))
        return {'input_tokens': input_tokens, 'output_reserve_tokens': output,
            'safety_tokens': self.safety, 'context_window_tokens': self.window,
            'count_method': method or self.counter.method}

    def image_tokens(self, images):
        import base64
        import io
        from PIL import Image
        total = 0
        for image in images:
            with Image.open(io.BytesIO(base64.b64decode(image.split(',', 1)[1], validate=True))) as value:
                width, height = value.size
            patches = math.ceil(width / self.options['image_patch_pixels']) * math.ceil(height / self.options['image_patch_pixels'])
            total += max(self.options['image_tokens_minimum'], patches * self.options['image_tokens_per_patch'])
        return total

    def estimate(self, system, text, images=()):
        # Base64 is not language input; vision is accounted for separately.
        return self.counter.count(system) + self.counter.count(text) + self.image_tokens(images) + 128


def bounded_items(items, limit, *, counter=None):
    counter = counter or TokenCounter()
    selected = []
    used = 2
    for item in items:
        size = counter.count(item) + 1
        if used + size > limit:
            continue
        selected.append(item)
        used += size
    return selected
