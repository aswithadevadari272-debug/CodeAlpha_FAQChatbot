# FAQ Chatbot

**CodeAlpha AI Internship – Task 2**

A chatbot that answers frequently asked questions using Natural Language Processing. It matches the user's question to the closest FAQ using TF-IDF and cosine similarity, and replies with the stored answer. It comes with a simple chat window built in Tkinter.

## Features
- Understands differently worded questions (e.g. "how do I send back an item?" matches the return FAQ)
- Handles greetings, thanks and goodbyes
- Answers general questions too: if no FAQ matches, it looks the topic up on Wikipedia (needs internet)
- Gives a polite fallback reply when it finds nothing
- Easy to customise: edit `faqs.json` to use your own questions and answers
- Chat window (GUI) and terminal mode (`--cli`)

## Tech Stack
- Python 3
- scikit-learn (TF-IDF, cosine similarity)
- Tkinter

## How to Run
```bash
git clone https://github.com/<your-username>/CodeAlpha_FAQChatbot.git
cd CodeAlpha_FAQChatbot
pip install -r requirements.txt
python chatbot.py          # chat window
python chatbot.py --cli    # terminal mode
```

## How It Works
1. **Preprocessing:** each question is lowercased and cleaned of punctuation.
2. **Vectorisation:** TF-IDF turns every FAQ question into a numeric vector (stop words removed, single words and word pairs used).
3. **Matching:** the user's message is converted the same way, and cosine similarity finds the closest FAQ.
4. **Answering:** if the best score is above the threshold (0.15), that FAQ's answer is returned.
5. **General questions:** otherwise the question is searched on Wikipedia and a short summary is returned. If nothing is found, the bot says it didn't understand.

## Author
Aswitha Devadari
