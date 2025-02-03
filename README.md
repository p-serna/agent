# Deep Open Dive

This is a small repo for an LLM Agent using OpenAI library. It can be used as an alternative to deep research or other agentic workflows. I use it with ollama by fixing in a .env file `OPENAI_BASE_URL=http://localhost:11434/v1`. The implementation uses tools included in src/tools.py

To run it, install requirements
```
pip install -r requirements.txt
```
And run for example:
```
python -m src.agent task
```
where task is a file containing the instructions. If you want to use interactive mode run
```
python -m src.agent
```

## TODO
- [ ] Proper python interpreter
  - Issue: security
