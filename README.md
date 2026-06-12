# Institutional Quant Terminal — Cloud-Ready Platform

Multi-agent trading research desk (Macro Economist, Quant Strategist,
Risk Structurer) behind an authenticated, rate-limited Streamlit front end.

## Layout
```
quant-terminal/
├── app.py                      # Streamlit entry point (auth wall first)
├── make_user.py                # bcrypt user provisioning CLI
├── config/
│   ├── security.py             # hashing, sessions, RBAC, sanitization, masking
│   └── auth_config.yaml        # bcrypt hashes + cookie key (gitignored)
├── agents/
│   ├── base.py                 # AgentReport contract + safe LLM JSON parsing
│   ├── macro_analyst.py        # regime classification (FRED + headlines)
│   ├── quant_strategist.py     # z-scores, vol breaks, divergences
│   ├── risk_structurer.py      # stops from volume profile, Kelly/VaR sizing
│   └── orchestrator.py         # async committee + bounded debate round
├── core/math_engine.py         # pure quant kernels (unit-testable)
├── services/
│   ├── api_client.py           # HTTPS-only client, timeouts, retries, vendors
│   ├── rate_limiter.py         # sliding window (Redis-ready interface)
│   └── demo_data.py            # offline synthetic data for dev/CI
└── docs/DEPLOYMENT.md          # TLS 1.3 proxy, headers, go-live checklist
```

## Quick start (local)
> Seule SESSION_SIGNING_KEY est obligatoire. Les données de marché (Yahoo Finance) et les actualités (RSS) sont gratuites et sans clé.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # fill in keys
python -c "import secrets;print(secrets.token_hex(32))"   # -> SESSION_SIGNING_KEY
python make_user.py admin admin            # paste output into auth_config.yaml
streamlit run app.py --server.address 127.0.0.1
```

## Security model (summary)
- Fail-closed boot: missing/weak secrets abort startup.
- bcrypt (cost 12) passwords; HMAC-SHA256 signed session claims; RBAC on
  every action; constant-time comparisons.
- Allowlist ticker validation kills injection at the only free-text input.
- Sliding-window rate limits per user per resource (LLM budget protected).
- HTTPS-only outbound client, strict timeouts, scrubbed error logging,
  RedactingFilter so secrets cannot reach logs even by accident.
- No order execution path on the public app — research signals only.

> Output is quantitative research, not investment advice. Backtest before
> risking capital; the sizing priors are placeholders to calibrate.
