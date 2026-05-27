# Cloudflare in front of the coord — bandwidth scaler

**What this buys**: Cloudflare caches each round's `/state?round=N` response at
the edge. Coord serves it **once per round**; every subsequent worker pull
in any region hits the closest Cloudflare PoP instead of the VPS. With 4
active workers per round and Cloudflare's global edge, our VPS egress for
state drops by **~75-95%**.

**When to do this**: as soon as monthly bandwidth becomes a constraint, OR
proactively before cohort grows past ~5 workers.

**Time required**: 10 minutes of clicking + 5 minutes to 24 hours of DNS
propagation (usually 5-15 min in practice).

**Cost**: Free tier covers everything we need.

---

## Prerequisites

The code side is already done (see commits `6c67a65` + this one). Coord
serves `/state?round=N` with `Cache-Control: public, max-age=86400,
immutable`. Worker passes `?round=N` on every `/state` pull. Both are
production-ready — they just need Cloudflare in the path.

## Steps

### 1. Sign up

[cloudflare.com/sign-up](https://www.cloudflare.com/sign-up) — free
account. No credit card required for the free plan.

### 2. Add the zone

- Dashboard → **Add a Site** → enter `planetbass.de` → Free plan.
- Cloudflare scans existing DNS records (might pull from your registrar).
  Verify these are listed:
  - `A   dllm.planetbass.de   159.195.34.222`  (proxied 🟠 — clouds means
    "behind Cloudflare", NOT just DNS)
  - any others you have for `planetbass.de` — set those grey clouds
    (DNS-only) unless you want them cached too.

Critical: the `dllm` record's proxy toggle must be **orange**. Grey means
Cloudflare resolves DNS but the user's connection goes direct to your
origin — no caching, no CDN benefit.

### 3. Change name servers

Cloudflare gives you 2 name servers like:
```
gail.ns.cloudflare.com
seth.ns.cloudflare.com
```

Log into your domain registrar (wherever you bought `planetbass.de` — INWX,
Strato, GoDaddy, etc.) and change the name servers from your registrar's
defaults to Cloudflare's pair.

Propagation: typically 5-15 minutes, occasionally up to 24 hours. Watch
Cloudflare's dashboard — it'll switch from "Pending" to "Active".

### 4. Verify SSL still works

Cloudflare sits between the client and your origin. Two SSL legs:

- **Client ↔ Cloudflare**: Cloudflare's free SSL (Universal SSL),
  auto-issued. Just works.
- **Cloudflare ↔ Origin**: needs configuration. Two options:
  - **Flexible** (default): Cloudflare talks HTTPS to clients, plain HTTP
    to origin. *Don't pick this* — your origin nginx redirects 80 → 443,
    so you'd get a redirect loop.
  - **Full (strict)** (recommended): Cloudflare validates your origin's
    Let's Encrypt cert. *This is what we want.*

Set this at **SSL/TLS → Overview → Encryption mode → Full (strict)**.

Test: `curl -fsS https://dllm.planetbass.de/health` should still work.

### 5. Add the cache rule

By default Cloudflare doesn't cache "uncommon" content types (anything not
.jpg/.css/.js/.pdf/etc.). Our `/state` endpoint serves
`application/octet-stream` — not in the default list. We need a rule.

**Caches → Cache Rules → Create rule**:

- **Rule name**: `cache /state per round`
- **Match**: `URI Path starts with /state`
- **Cache eligibility**: `Eligible for cache`
- **Edge TTL**: `Respect origin TTL` (it'll honor our `max-age=86400`)
- **Cache key**: include query string (default — make sure `round=N` is
  part of the cache key so each round caches separately)

Save → deploys instantly.

### 6. Verify it's working

After DNS propagates + cache rule is active, run:

```bash
curl -I "https://dllm.planetbass.de/state?round=1"  # first pull
curl -I "https://dllm.planetbass.de/state?round=1"  # repeat
```

Look at the `cf-cache-status` response header:
- First call: `MISS` or `EXPIRED` — origin served it.
- Second call: `HIT` — Cloudflare served it from cache.

Or fetch `/state?round=N` from two different regions and watch the
coord's bandwidth: the second region shouldn't trigger any origin traffic.

## Operational notes

- Cache fills the **first time** any worker requests a given `?round=N`.
  Subsequent workers in the same round hit cache.
- The 1-hour worker_inactive_timeout means stale registrations can sit for
  up to an hour. Those don't pull state. The cap on n_workers means at
  most 4 active workers per round so the worst case is 4 origin hits per
  round (and Cloudflare's anycast usually deduplicates those too).
- Cloudflare's free tier doesn't cache responses larger than 512 MB by
  default — `/state` for our 300M model is ~600 MB, just over the limit.
  Workaround: upgrade to Cloudflare Pro ($20/mo, raises to 1 GB) OR shard
  the state into 256 MB chunks served by separate cache-friendly URLs.
  For Phase 0–1 we accept the limit and revisit when needed.

## Rollback

If anything breaks:

1. **Quick rollback**: in Cloudflare → DNS → toggle the `dllm` record
   to **DNS-only (grey cloud)**. Traffic goes direct to origin, no caching,
   no CDN involvement. Effect within ~1 minute.
2. **Full rollback**: change name servers back at the registrar. ~5-30 min.

Code keeps working in all cases — the `?round=N` URL and 409 mismatch
behaviour are origin-side semantics, fine without Cloudflare.
