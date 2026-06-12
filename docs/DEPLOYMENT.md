# DEPLOYMENT.md — Production Hardening Checklist

Streamlit must **never** face the internet directly. Terminate TLS at a
reverse proxy, and let the proxy own DDoS absorption, header hygiene, and
HTTP→HTTPS redirection. The in-app rate limiter is the *second* line of
defense (protects the LLM/data budget per authenticated user); the proxy and
your cloud provider's edge (Cloudflare/AWS WAF) are the first.

## 1. Nginx — TLS 1.3, HSTS, security headers

```nginx
server {
    listen 443 ssl;
    http2 on;
    server_name terminal.example.com;

    ssl_certificate     /etc/letsencrypt/live/terminal.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/terminal.example.com/privkey.pem;
    ssl_protocols       TLSv1.3;            # 1.3 only, per policy
    ssl_prefer_server_ciphers off;

    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header Referrer-Policy no-referrer always;

    # Edge rate limit (per-IP) BEFORE traffic reaches the app:
    limit_req zone=app burst=20 nodelay;

    location / {
        proxy_pass http://127.0.0.1:8501;     # Streamlit bound to localhost only
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;   # Streamlit needs WebSockets
        proxy_set_header Connection "upgrade";
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_read_timeout 300s;
    }
}

# In the http{} block:
# limit_req_zone $binary_remote_addr zone=app:10m rate=10r/s;
```

Run Streamlit bound to loopback so it is unreachable except via the proxy:

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8501 \
  --server.enableXsrfProtection true
```

## 2. Binance WebSocket feed (market data)

Use `wss://stream.binance.com:9443/ws/<symbol>@trade` — the `wss://` scheme
enforces TLS. Read-only market streams need **no API key**; do not ship
trading keys to this server at all unless order execution is in scope. If it
ever is: create keys with *read + trade only* (withdrawals disabled) and IP
allowlisting on the Binance side.

```python
# services/binance_ws.py (sketch)
import ssl, websockets
ctx = ssl.create_default_context()
ctx.minimum_version = ssl.TLSVersion.TLSv1_2
async with websockets.connect(
    "wss://stream.binance.com:9443/ws/btcusdt@aggTrade", ssl=ctx,
    ping_interval=20, ping_timeout=10, max_size=2**20,   # bound message size
) as ws:
    ...
```

## 3. Secrets lifecycle

- Local dev: `.env` (gitignored). Production: the platform's secret store
  (AWS Secrets Manager / GCP Secret Manager / Docker secrets), injected as
  env vars — `load_environment()` reads either transparently.
- `config/auth_config.yaml` contains bcrypt hashes + the cookie signing key;
  it is gitignored. Provision users with `python make_user.py <name> <role>`.
- Rotate `SESSION_SIGNING_KEY` and the cookie key on any suspicion of
  compromise; all sessions invalidate automatically.
- The Admin tab's secrets report calls `assert_logs_are_masked()` — it will
  *raise* if any configured key would render unmasked.

## 4. Container & runtime

- Run as a non-root user in the container; read-only filesystem where possible.
- `pip install -r requirements.txt --require-hashes` once you pin hashes.
- Healthcheck on `/_stcore/health`; restart policy `on-failure`.
- Egress allowlist at the firewall: api.anthropic.com, api.stlouisfed.org,
  www.alphavantage.co, stream.binance.com — and nothing else.

## 5. Go-live checklist

- [ ] `.env` populated; `SESSION_SIGNING_KEY` ≥ 64 hex chars
- [ ] Default `admin` placeholder hash in auth_config.yaml replaced
- [ ] Streamlit bound to 127.0.0.1; only :443 exposed publicly
- [ ] TLS 1.3 verified (`testssl.sh terminal.example.com`)
- [ ] Edge rate limiting + WAF enabled at the proxy/CDN
- [ ] Log review: confirm RedactingFilter output shows masked keys only
- [ ] Backtest your committee's signals before sizing real capital — the
      win-rate/payoff priors in `risk_structurer.py` are placeholders

## 6. Scope note

This platform produces *research signals*, not investment advice, and ships
with **no order-execution path** by design. Keeping execution out of the
public-facing app is itself a security control: a compromised dashboard can
leak analysis, but it cannot move funds.
