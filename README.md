---
title: Telegram Fake News Bot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: main.py
pinned: false
---

# Telegram Fake News Detection Bot (Webhook API)

This Hugging Face Space hosts a FastAPI app that connects your Telegram bot to a fake news classification model hosted on Hugging Face.

- It uses `os.getenv` to access secrets securely.
- Set your `TELEGRAM_BOT_TOKEN` and `HF_SPACE_API` under Settings → Secrets after deploying.
- You can then set the webhook using the Telegram Bot API.

This solution runs 24/7 for free and connects your resume-linked Telegram bot to your Hugging Face inference model.
