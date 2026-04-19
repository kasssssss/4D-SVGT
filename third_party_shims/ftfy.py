"""Minimal local shim for environments without ftfy."""


def fix_text(text, *args, **kwargs):
    return text
