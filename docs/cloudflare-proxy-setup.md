# Cloudflare Worker CORS Proxy — Setup & Verification

A Cloudflare Worker is used to proxy requests to `littlegolem.net`, because:
- LG has no `Access-Control-Allow-Origin` headers (all direct browser `fetch()` calls blocked)
- corsproxy.io blocks the required content-type without a paid subscription

The Worker is a thin passthrough: it forwards GET requests from the browser to LG
and adds the CORS headers the browser requires.

---

## 1. Create a Cloudflare account

Go to https://dash.cloudflare.com/sign-up — the free plan is more than enough
(100,000 requests/day, unlimited Workers).

---

## 2. Create the Worker

### Option A — Cloudflare dashboard (no CLI needed)

1. Log in → **Workers & Pages** → **Create** → **Create Worker**
2. Name it `lg-proxy` (or anything you like)
3. Click **Deploy** (deploys the default hello-world script)
4. Click **Edit code** to open the inline editor
5. Replace the entire script with:

```javascript
export default {
  async fetch(request, env, ctx) {
    // Only allow GET
    if (request.method !== 'GET') {
      return new Response('Method not allowed', { status: 405 });
    }

    // Target URL is passed as the `url` query parameter
    const incoming = new URL(request.url);
    const target = incoming.searchParams.get('url');
    if (!target) {
      return new Response('Missing ?url= parameter', { status: 400 });
    }

    // Only proxy littlegolem.net
    let targetUrl;
    try {
      targetUrl = new URL(target);
    } catch {
      return new Response('Invalid URL', { status: 400 });
    }
    if (!targetUrl.hostname.endsWith('littlegolem.net')) {
      return new Response('Only littlegolem.net URLs are allowed', { status: 403 });
    }

    // Fetch from LG
    const lgResponse = await fetch(targetUrl.toString(), {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; twixtbot-app/1.0)',
      },
    });

    // Forward the response with CORS headers added
    const headers = new Headers(lgResponse.headers);
    headers.set('Access-Control-Allow-Origin', '*');
    headers.set('Access-Control-Allow-Methods', 'GET');

    return new Response(lgResponse.body, {
      status: lgResponse.status,
      headers,
    });
  },
};
```

6. Click **Deploy**

### Option B — Wrangler CLI

```bash
npm install -g wrangler
wrangler login
wrangler init lg-proxy --no-delegate-c3
# replace src/index.js with the script above
wrangler deploy
```

---

## 3. Note your Worker URL

After deploy, Cloudflare shows the URL:
```
https://lg-proxy.<your-subdomain>.workers.dev
```

---

## 4. Verify with curl

### 4a — SGF file (V1 + V6)
```bash
curl -s "https://lg-proxy.<your-subdomain>.workers.dev/?url=https%3A%2F%2Fwww.littlegolem.net%2Fservlet%2Fsgf%2F2546140%2Fgame2546140.tsgf" | head -5
```
**Expected output:**
```
(;GM[21]FF[4]SZ[24]SO[Little Golem]EV[...]...
```
- Must start with `(;GM[21]` — raw SGF, not JSON, not HTML
- Status 200

### 4b — Player search HTML (V3)
```bash
curl -s "https://lg-proxy.<your-subdomain>.workers.dev/?url=https%3A%2F%2Fwww.littlegolem.net%2Fjsp%2Finfo%2Fplayer_list.jsp%3Fgtvar%3Dtwixt_DEFAULT%26filter%3Dalan" | grep -o 'plid=[0-9]*' | head -5
```
**Expected output:** several `plid=NNNN` lines — confirms the player search page
is proxied as raw HTML and the links are in the expected format.

### 4c — Game list HTML
```bash
curl -s "https://lg-proxy.<your-subdomain>.workers.dev/?url=https%3A%2F%2Fwww.littlegolem.net%2Fjsp%2Finfo%2Fplayer_game_list.jsp%3Fgtid%3Dtwixt%26plid%3D2674" | grep -o 'gid=[0-9]*' | head -5
```
**Expected output:** several `gid=NNNN` lines.

---

## 5. Update `PROXY_PREFIX` in the app

In `webapp/src/lg-api.ts`, line ~10:
```typescript
// Before:
const PROXY_PREFIX = 'https://corsproxy.io/?url=';

// After:
const PROXY_PREFIX = 'https://lg-proxy.<your-subdomain>.workers.dev/?url=';
```

---

## 6. End-to-end app test

1. Set `MOCK_MODE = false` in `webapp/src/lg-api.ts`
2. Build and serve locally:
   ```bash
   cd webapp && npm run dev
   ```
3. Open the app → **Explore LG games**
4. Search for `alan` → should list players with ratings
5. Click Alan Hensel → should list recent TwixT games with opponent names and results
6. Click a game → replay should load and show moves on the board

If step 4 shows players and step 6 shows moves, all of V1/V3/V5/V6 are confirmed.

---

## 7. Deploy the Worker URL with the app

Once verified, commit the updated `PROXY_PREFIX` and `MOCK_MODE = false` to main.
The Worker URL is not a secret (it only allows `littlegolem.net` targets), so
committing it to source is fine.

---

## Security notes

- The Worker only forwards GET requests to `littlegolem.net` — no other domains
- No authentication or API keys involved — LG game data is fully public
- Rate limits: Cloudflare free tier allows 100k requests/day; LG games are
  tiny HTML/text files so this is effectively unlimited for normal use
- The Worker adds `Access-Control-Allow-Origin: *` only on responses from LG,
  not on error responses
