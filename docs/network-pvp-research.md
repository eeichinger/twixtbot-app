# Research: P2P Multiplayer Options for TwixT PWA

**Date:** 2026-04-04  
**Context:** Exploring how two players, each with the PWA installed on their own phone or
tablet, could play TwixT against each other peer-to-peer — without a dedicated server.

> **Feature status and priority tracked in `docs/planned-features.md` (section 7, items R1–R4).
> This document is the research reference; do not duplicate status here.**

---

## Requirements & constraints

- Two phones/tablets, each running the PWA independently
- No dedicated server available (app hosted on GitHub Pages, static only)
- Target scenarios: same WLAN, or physically nearby (Bluetooth)
- **Hard requirement: iOS Safari 18+** must be supported
- App is cross-origin-isolated (`COOP: same-origin` + `COEP: require-corp`) for
  SharedArrayBuffer — any solution must be compatible with this

---

## What the browser actually offers

| API | iOS Safari | Android Chrome | Peer-to-peer capable? |
|-----|-----------|----------------|----------------------|
| WebRTC DataChannel | ✅ iOS 11+ | ✅ | ✅ yes — after a signaling handshake |
| Web Bluetooth | ❌ not in Safari | ✅ Chrome 57+ | ❌ connects to BLE *peripherals* only, not browser↔browser |
| Web NFC | ❌ | ✅ Chrome 89+ | ❌ pairing tap only; 10 cm range, 424 B/s |
| Direct Sockets / mDNS | ❌ | Chrome IWA only | ❌ requires Isolated Web App, not GitHub Pages |
| BroadcastChannel | ✅ | ✅ | same-device tabs only |

**Conclusion:** WebRTC DataChannel is the only viable cross-platform transport.

- **Web Bluetooth** is eliminated by iOS Safari's lack of support. Even on Android,
  the browser-side API connects only to BLE *peripherals* (headphones, sensors) —
  two browser tabs cannot connect to each other over BLE.
- **Web NFC** has no iOS support and is unsuitable for game data (range, throughput).
- **Local mDNS / LAN discovery** — there is no browser API for this. Chrome's
  Direct Sockets API (which could do UDP multicast) is restricted to Isolated Web Apps
  and will not work on a GitHub Pages deployment.

---

## The signaling problem

WebRTC is genuinely peer-to-peer *after* the connection is established, but the initial
SDP offer/answer + ICE candidate exchange requires an out-of-band signaling channel.
Something must relay approximately 1 KB of text between the two clients before the
DataChannel opens. That "something" can range from a cloud service to a QR code.

### iOS-specific constraint: TURN relay required

iOS Safari does not expose local host ICE candidates for security reasons.
This means STUN-only ICE fails on iOS — a TURN relay server is required even when
both devices are on the same WiFi network. Free TURN servers are available:

- **OpenRelay (metered.ca)** — `turns:openrelay.metered.ca:443` — free, no sign-up
- Self-host coturn on any free VPS (optional, for full control)

### COOP/COEP compatibility

The app injects `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` on all responses via the Service Worker.
All candidates below either bundle their SDK via npm (code becomes same-origin after
Vite build) or use WebSocket connections (not intercepted by the Service Worker).
No COEP conflict in practice.

---

## Solution candidates (ranked by implementation simplicity)

---

### Candidate 1 — PeerJS + PeerJS Cloud  ⭐ simplest

**How it works:**
1. Player 1 opens "Host game" → `new Peer()` → receives a short alphanumeric peer ID
   (e.g. `ax7k`) from the PeerJS Cloud signaling server
2. Player 1 shares the ID with Player 2 (spoken aloud, or auto-appended to a share URL)
3. Player 2 opens "Join game", enters the ID → `peer.connect(id)` → DataChannel opens
4. Game moves are sent as JSON frames over the DataChannel; PeerJS Cloud is no longer
   involved after the connection is established

**Peer discovery:** A 4-character code shared verbally or via URL. No QR scanning needed.

**Implementation cost:**
- `npm install peerjs` (~180 KB gzip after tree-shaking)
- ~60–80 lines of TypeScript for a `RemoteConnection` manager class
- Free PeerJS Cloud signaling server — no account, no deploy, no config
- Add OpenRelay TURN credentials to `RTCPeerConnection({ iceServers })` for iOS

**Pros:**
- Lowest code volume of any option
- PeerJS abstracts the entire WebRTC and signaling stack
- No infrastructure to deploy or maintain

**Cons:**
- Depends on PeerJS Cloud uptime (public server, no SLA)
- If reliability becomes a concern, `peerserver` can be self-hosted on any free Node
  host (Railway, Fly.io, Render free tier) with a one-line config change

---

### Candidate 2 — p2pcf (Cloudflare Workers + WebRTC)

**How it works:**
1. Deploy a ~20-line Cloudflare Worker that acts as an HTTP-polling SDP store
2. `npm install p2pcf`; share a room code; `p2pcf` handles signaling via the Worker
3. After the DataChannel opens, the Worker is no longer in the data path

**Peer discovery:** Short room code shared verbally or via URL (same UX as Candidate 1).

**Implementation cost:**
- One-time: deploy Cloudflare Worker with `wrangler deploy` (free, ~30 lines of JS)
- ~80–100 lines of TypeScript in the app
- Free Cloudflare tier (1M requests/month)

**Pros:**
- No third-party cloud lock-in beyond Cloudflare (which you already use implicitly
  via GitHub Pages's CDN)
- The Worker is fully under your control
- p2pcf bundle is tiny (~30 KB)

**Cons:**
- Requires a one-time Cloudflare account setup and Worker deploy
- p2pcf is less actively maintained than PeerJS

---

### Candidate 3 — WebRTC + Firebase Realtime Database signaling

**How it works:**
1. Firebase RTDB stores `{ offer, answer, offerCandidates[], answerCandidates[] }` keyed
   by a room ID
2. Each player subscribes via `onValue()` for the other's SDP and ICE candidates
3. After signaling completes, game data flows over the DataChannel; Firebase is idle

**Peer discovery:** Share a room code, or auto-generate a URL with `?room=xxxx`.

**Implementation cost:**
- Add Firebase SDK (only `firebase/app` + `firebase/database`, ~40 KB gzip bundled)
- ~150–180 lines of TypeScript for the signaling layer
- Free Firebase tier (100 concurrent connections, 1 GB/month)

**Pros:**
- Rock-solid, widely-deployed signaling infrastructure
- Real-time push (no polling latency)
- Easy to extend later with lobby, presence, or chat

**Cons:**
- Requires a Google/Firebase account and project setup
- Larger dependency than PeerJS or p2pcf
- Game data (SDP during setup) passes through Google infrastructure

---

### Candidate 4 — Manual QR-code SDP exchange (truly serverless)

**How it works:**
1. Player 1 generates an SDP offer → encodes as a QR code shown on screen
2. Player 2 scans the QR → app parses the offer, generates an answer → shows answer QR
3. Player 1 scans the answer QR → connection opens

**Peer discovery:** Physical QR scan — requires both devices to be in the same room.

**Implementation cost:**
- ~150 lines of TypeScript + a QR encode/decode library (e.g. `qrcode` + `jsqr`)
- SDP for a data-channel-only connection is ~400–600 bytes — fits in a standard QR code
- ICE candidate trickle must be disabled (gather all candidates before encoding QR);
  this adds 1–3 seconds of latency at connection setup

**Pros:**
- Zero server dependency whatsoever

**Cons:**
- Requires two QR scans (one per direction), poor UX
- Without TURN, iOS ICE will fail (TURN credentials can't be embedded without
  exposing them in the QR — a security concern)
- Not recommended for production use; interesting as a fallback or demo

---

## Recommendation

**Candidate 1 (PeerJS)** is the right starting point:
- Fastest to implement (~60–80 lines + npm install)
- No infrastructure to deploy
- Proven on iOS Safari
- Easily upgraded to a self-hosted PeerServer if reliability becomes a concern

**Candidate 2 (p2pcf)** is the right choice if you want zero dependency on any
third-party cloud service and are comfortable with a one-time Cloudflare Worker deploy.

Both require the OpenRelay TURN server for iOS. Add to `RTCPeerConnection`:

```typescript
iceServers: [
  { urls: 'stun:openrelay.metered.ca:80' },
  {
    urls: 'turns:openrelay.metered.ca:443',
    username: 'openrelayproject',
    credential: 'openrelayproject',
  },
]
```

---

## Estimated implementation scope

| Layer | Approx. lines of TS |
|-------|---------------------|
| `RemoteConnection` manager (offer/answer/ICE/state machine) | ~100 |
| Game message protocol (move, undo, resign, new-game sync) | ~50 |
| UI additions (host/join screen, room code display, status, reconnect) | ~80 |
| **Total** | **~230** |

### Integration with existing architecture

The existing codebase has three game modes: `pvc` (vs computer) and `pvp` (local two-player).
The remote mode would be a third `GameMode` value `pvp-remote`. The DataChannel message
format can mirror the existing worker message format `{ type, history }` so the game logic
layer (`main.ts`) needs minimal changes — it already handles move application and turn
management independently of transport.

The `game-mode.ts` helper `isHumanTurn()` and `turnStatusText()` already handle
the two-human-player case and would work unchanged for remote mode.
