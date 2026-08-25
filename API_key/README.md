# API key

The flow reads the LLM API key in this order (see `resolve_api_key` in
`macroflow/llm_openai.py`):

1. `openai.api_key` in the model YAML (supports `${VAR}` expansion)
2. `$OPENAI_API_KEY`
3. `openai.api_key_file` — a file containing only the key

To use the file route, copy the template and paste your key:

```bash
cp API_key/api_key_openai.md.example API_key/api_key_openai.md
```

`API_key/*` is git-ignored (except this README and the `.example` template),
so a real key cannot be committed by accident. The file route exists because
parallel candidate workers are spawned as separate processes that may not
inherit a shell-exported variable.
