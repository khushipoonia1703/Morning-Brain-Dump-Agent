# 🧠 Morning Brain Dumper

**Your brain is messy. Your day doesn't have to be.**

Morning Brain Dumper turns a messy voice dump into a structured **Notion checklist** — so instead of spending your morning organizing what’s in your head, you can just start doing.

### How it works

```text
🎙️ Voice Dump
     ↓
📝 Transcribe
     ↓
🧩 Split + Classify
     ↓
📚 Research what needs resources
     ↓
✅ Verify resources
     ↓
📓 Notion
```

Your rambling gets turned into three simple buckets:

- 📚 **Study** — things you need to learn
- ✅ **Tasks** — things you need to do
- ⏰ **Reminders** — things tied to a time

For study items, the research agent finds **one useful video + one article**, then verifies the URLs before they reach your final page.

### The interesting part

This isn't an “LLM does everything” project.

It's a **deterministic pipeline with one bounded agent inside it**.

The LLM decides **what to research and which tool to use**.  
The code decides **what is allowed, what is valid, and when the agent must stop**.

> **The agent makes decisions. The code enforces the rules.**

### Built with

**Python · OpenAI SDK · Tavily · Notion API**

### Why I built it

Because the hardest part of a busy morning isn't knowing what to do.

It's turning **“I have so much to do”** into **“I know exactly where to start.”**
