"""
CodeAlpha Internship - Task 2: Chatbot for FAQs
Answers frequently asked questions using NLP:
text cleaning -> TF-IDF vectors -> cosine similarity matching.
Questions that are not in the FAQ list are looked up on Wikipedia.
Run with GUI:  python chatbot.py
Run in terminal: python chatbot.py --cli
"""

import json
import os
import re
import sys
import threading
import urllib.parse
import urllib.request
import tkinter as tk
from tkinter import ttk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_FILE = os.path.join(BASE_DIR, "faqs.json")
THRESHOLD = 0.15  # minimum similarity needed to trust a match

GREETINGS = {"hi", "hello", "hey", "hii", "good morning", "good evening"}
GOODBYES = {"bye", "goodbye", "see you", "exit", "quit"}
THANKS = {"thanks", "thank you", "thx", "thanks a lot"}

FALLBACK = ("Sorry, I couldn't find an answer to that. "
            "Try rephrasing your question.")


def preprocess(text: str) -> str:
    """Lowercase, remove punctuation and extra spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


QUESTION_PREFIX = re.compile(
    r"^(what is|what are|what's|who is|who was|who are|tell me about|explain|"
    r"define|where is|when was|when did|how does|how do|why is|why do)\s+(an?\s+|the\s+)?"
)


def wikipedia_answer(query: str):
    """Look the question up on Wikipedia. Returns a short summary or None."""
    headers = {"User-Agent": "CodeAlphaFAQChatbot/1.0 (student project)"}

    def get_json(url):
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))

    term = QUESTION_PREFIX.sub("", query.lower().strip(" ?!."))
    search_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": term or query,
        "srlimit": 3, "format": "json",
    })
    for result in get_json(search_url)["query"]["search"]:
        title = result["title"]
        summary_url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
                       + urllib.parse.quote(title.replace(" ", "_")))
        data = get_json(summary_url)
        extract = data.get("extract", "")
        if data.get("type") == "disambiguation" or not extract:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", extract)
        return " ".join(sentences[:3]) + f"\n(Source: Wikipedia - {title})"
    return None


class FAQBot:
    def __init__(self, faq_path: str = FAQ_FILE):
        with open(faq_path, encoding="utf-8") as f:
            self.faqs = json.load(f)

        # Match against the question plus optional keywords (synonyms)
        questions = [preprocess(item["question"] + " " + item.get("keywords", ""))
                     for item in self.faqs]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(questions)

    def get_answer(self, user_input: str) -> str:
        cleaned = preprocess(user_input)

        if not cleaned:
            return "Please type a question."
        if cleaned in GREETINGS:
            return "Hello! How can I help you today?"
        if cleaned in THANKS:
            return "You're welcome! Anything else I can help with?"
        if cleaned in GOODBYES:
            return "Goodbye! Have a great day."

        user_vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(user_vec, self.matrix)[0]
        best = scores.argmax()

        if scores[best] >= THRESHOLD:
            return self.faqs[best]["answer"]

        # Not an FAQ: try to answer from Wikipedia
        try:
            answer = wikipedia_answer(user_input)
            if answer:
                return answer
        except Exception:
            return ("I couldn't reach the internet to look that up. "
                    "Please check your connection and try again.")
        return FALLBACK


class ChatApp:
    def __init__(self, root: tk.Tk, bot: FAQBot):
        self.root = root
        self.bot = bot
        root.title("FAQ Chatbot - CodeAlpha")
        root.geometry("600x540")
        root.minsize(480, 400)

        frame = ttk.Frame(root, padding=10)
        frame.pack(fill="both", expand=True)

        self.log = tk.Text(frame, wrap="word", state="disabled",
                           font=("Segoe UI", 11), bg="#fafafa")
        scroll = ttk.Scrollbar(frame, command=self.log.yview)
        self.log.config(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

        self.log.tag_config("user", foreground="#0b57d0", font=("Segoe UI", 11, "bold"))
        self.log.tag_config("bot", foreground="#1e1e1e")

        bottom = ttk.Frame(root, padding=(10, 0, 10, 10))
        bottom.pack(fill="x")
        self.entry = ttk.Entry(bottom, font=("Segoe UI", 11))
        self.entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.entry.bind("<Return>", self.send)
        self.entry.focus()
        ttk.Button(bottom, text="Send", command=self.send).pack(side="left", padx=(8, 0))

        self.add_message("Bot", "Hi! Ask me about orders, delivery, returns, payments "
                                "or your account.", "bot")

    def add_message(self, sender: str, text: str, tag: str):
        self.log.config(state="normal")
        self.log.insert("end", f"{sender}: ", tag)
        self.log.insert("end", text + "\n\n", "bot")
        self.log.config(state="disabled")
        self.log.see("end")

    def send(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self.add_message("You", text, "user")
        # Run in a thread so the window doesn't freeze during web lookups
        threading.Thread(target=self.reply, args=(text,), daemon=True).start()

    def reply(self, text: str):
        answer = self.bot.get_answer(text)
        self.root.after(0, self.add_message, "Bot", answer, "bot")


def run_cli(bot: FAQBot):
    print("FAQ Chatbot (type 'quit' to exit)")
    while True:
        text = input("You: ")
        print("Bot:", bot.get_answer(text))
        if preprocess(text) in GOODBYES:
            break


if __name__ == "__main__":
    bot = FAQBot()
    if "--cli" in sys.argv:
        run_cli(bot)
    else:
        root = tk.Tk()
        ChatApp(root, bot)
        root.mainloop()
