# 🌎 DEET — AI Language Learning Through Conversation

> **DEET** = **D**aniel · **E**dison · **E**van · **T**engis

![Node.js](https://img.shields.io/badge/Node.js-339933?logo=nodedotjs&logoColor=white)
![Express](https://img.shields.io/badge/Express-000000?logo=express&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?logo=google&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-02569B?logo=flutter&logoColor=white)

A summer startup project built with friends — the idea was to help people practice a new language the way you actually get good at it: by having real conversations. Instead of drilling flashcards or memorizing grammar rules, DEET drops you into a casual chat with an AI persona tuned to your level, speaking simply in your target language and always providing translations alongside.

---

## The Concept

Most language apps teach you *about* a language. DEET was designed to make you *use* it. The core loop:

1. Pick a target language
2. Get matched with an AI conversation partner (e.g., "Enrique" — a native Spanish speaker from Spain)
3. Have a natural back-and-forth, with the AI using simple vocabulary and showing English translations in parentheses so you never feel lost
4. Every message has a 🔊 read-aloud button so you can hear native-sounding pronunciation

The AI remembers your conversation history throughout the session, so exchanges feel continuous rather than isolated Q&A.

---

## What Was Built

### Web Prototype (`Web_App/`)

A working chat interface backed by **Google Gemini Pro**. The server maintains per-session chat history and injects a system prompt that configures the AI persona — Enrique speaks only in simple Spanish phrases, always with English translations in parentheses, keeping things accessible for beginners.

| Technology | Purpose |
|---|---|
| Node.js + Express | Web server |
| Google Gemini Pro (`@google/generative-ai`) | Conversational AI |
| Handlebars (hbs) | Server-side templating |
| express-session | Per-user chat history |
| Web Speech API | Browser-native read-aloud TTS |

### TTS Exploration (`Web_App/speech_test.py`)

Also explored **Suno Bark** — a transformer-based text-to-speech model with multilingual support and natural-sounding output. Tested generating Spanish and Chinese speech with different voice presets. The browser's Web Speech API was used in the final prototype for simplicity, but Bark showed promise for a more polished voice experience.

### Mobile App (`flutter_application_1/`)

Early-stage **Flutter** app targeting iOS and Android — got as far as a login screen scaffold before the project paused. The plan was a mobile-first experience with push notifications for daily conversation prompts.

---

## Getting Started

### Prerequisites
- Node.js 16+
- A free [Google Gemini API key](https://aistudio.google.com/)

### Setup

```bash
cd Web_App
npm install
cp .env.example .env
# Add your Gemini API key to .env
```

### Run

```bash
node server.js
```

Open `http://localhost:3000` — Enrique will be waiting.

---

### TTS Experiment (optional)

```bash
pip install git+https://github.com/suno-ai/bark.git
pip install git+https://github.com/huggingface/transformers.git
python speech_test.py
```

Outputs `bark_out.wav`. Supported languages listed [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c).

---

## Team

| Member | Role |
|---|---|
| Daniel Sun ([@dsun03](https://github.com/dsun03)) | Frontend |
| Edison Chen ([@ebc5802](https://github.com/ebc5802)) | Backend / AI integration |
| Evan Wang ([LinkedIn](https://www.linkedin.com/in/evan-wang-696bbb276/)) | Ideation |
| Tengis Otgonbaatar ([LinkedIn](https://www.linkedin.com/in/otengis/)) | Backend |
