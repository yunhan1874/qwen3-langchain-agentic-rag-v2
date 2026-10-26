# Security

Do not commit API keys, model weights, medical source data, generated indexes, or local `.env` files.

Use `.env.example` as the configuration template and inject real values through environment variables.
If a credential is found in Git history or a shared copy, revoke it immediately before removing it from the repository.

The Web UI renders model and retrieval output. Treat all corpus content as untrusted, keep the demo behind appropriate access controls, and do not expose a real medical corpus through a public Gradio share link.
