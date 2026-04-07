# TwixTBot-App

A Progressive Web App (PWA) that lets you play the board game **TwixT** against a
neural-network AI — or against another human on the same device — directly in your
mobile browser, with no installation required.

**Play now:** https://eeichinger.github.io/twixtbot-app/

---

## How to Play

### Open the app

Navigate to https://eeichinger.github.io/twixtbot-app/ in your browser.
The app loads entirely offline after the first visit.

### Install as a PWA (optional)

- **iOS Safari:** tap the Share button → "Add to Home Screen"
- **Android Chrome:** tap the three-dot menu → "Add to Home Screen" / "Install app"
- **Desktop Chrome/Edge:** click the install icon in the address bar

Once installed, the app works offline and launches like a native app.

### Choose a game mode

| Mode | Description |
|------|-------------|
| **vs Computer** | You play Black; the AI plays White |
| **vs Player** | Two humans take turns on the same screen |

Tap **Start Game** to begin.

### Making moves

- **Tap** any empty intersection to place your peg
- On touch screens, **drag** your finger to preview the peg before placing — release to confirm
- Bridges (links) between your pegs are drawn automatically when they are valid

### In-game controls

| Control | Action |
|---------|--------|
| **Undo** | Undo your last move (in vs-Computer mode, undoes both your move and the AI's) |
| **AI move** | Let the AI play your move for you (vs-Computer mode only) |
| **Think time selector** | Set how long the AI thinks: 5 s / 10 s / 30 s / 60 s |
| **New** | Return to the title screen |

### AI think time

More time = stronger play. The default is 10 seconds. On slower devices or if you
prefer a faster game, 5 seconds still produces strong moves.

---

## TwixT — Brief Rules

TwixT is a two-player connection game played on a 24×24 grid.

- **Black** connects top ↔ bottom; **White** connects left ↔ right
- Players take turns placing one peg per turn on any empty intersection
- After placing a peg, bridges (knight's-move links) are automatically added to all
  friendly pegs they can reach, provided no existing bridge would cross them
- **Swap rule:** White may swap the first Black move instead of placing a new peg
  (makes the opening fair regardless of which colour moves first)
- The first player to connect their two opposite borders wins
- Bridges cannot cross — this creates the tactical tension of the game

Full rules: https://en.wikipedia.org/wiki/TwixT

---

## Features

### Game modes
- **vs Computer** — play against a neural-network AI (MCTS + ONNX Runtime)
- **vs Player** — two humans take turns on the same device

### Gameplay
- Tap or click any intersection to place a peg
- **Touch drag-to-preview** — drag your finger to see where the peg will land, release to confirm
- Bridges are drawn automatically whenever they are valid (no crossing rule enforced)
- **Swap rule** — on the AI's opening move you can swap colours to equalise first-mover advantage
- **"AI move" button** — delegate any of your turns to the AI for a hint or to autopilot a move
- **Undo** — in vs-Computer mode undoes both your move and the AI's reply; in vs-Player undoes one move
- **Adjustable AI think time** — 5 s / 10 s / 15 s / 25 s / 30 s / 45 s / 60 s (choice persisted across sessions)
- **Win detection** with the winning bridge path highlighted in purple
- **Draw detection** when no legal moves remain

### Game import / export
- **Export** the current game as a `.tsgf` file to your device
- **LittleGolem explorer** — search for any LittleGolem player by name, browse their finished TwixT PP games, and open any game in the replay viewer
- Fetch a specific LittleGolem game directly by entering its numeric game ID
- **Download** any LittleGolem game as a `.tsgf` file
- **Paste or upload** a `.tsgf` file from your device to replay it

### Replay viewer
- Step through any game move by move (⏮ ← → ⏭ buttons)
- Keyboard navigation: arrow keys step one move, Home/End jump to start/end

### PWA / platform
- **Fully offline** after the first visit — Service Worker caches all app assets and the AI model
- **Installable** on iOS, Android, and desktop — no App Store required; launches like a native app
- Works on iOS Safari, Android Chrome, and all modern desktop browsers
- **Automatic updates** — new versions activate silently while you are on the intro screen, without interrupting an active game

---

## Attribution

This project is a fork of [**twixtbot**](https://github.com/BonyJordan/twixtbot) by
**Jordan Lampe** (MIT License, 2019). The original project is a Python 2 +
TensorFlow 1.12 command-line AI that uses MCTS and a convolutional neural network
trained by self-play. The trained model weights (`models/six-917000`) and the core
game logic (`src/twixt.py`) originate from Jordan's work.

The entire PWA — TypeScript port of the game logic, MCTS engine, ONNX Runtime Web
integration, Service Worker, Canvas UI, touch controls, iOS memory optimisations,
and diagnostic tooling — was designed and built by
[**Claude**](https://claude.ai) (Anthropic's AI assistant).

---

## Developer Notes

### Local development

```bash
cd webapp
npm install
npm run dev        # Vite dev server at http://localhost:5173
```

### Production build

```bash
cd webapp
npm run build      # Output in webapp/dist/
```

The `prebuild` script copies the required ONNX Runtime WASM files into `public/`
automatically.

### Project structure

```
src/               Original Python/TensorFlow bot (Jordan Lampe)
models/            Trained TF1 model weights
webapp/            TypeScript PWA (built by Claude)
  src/
    main.ts        App shell, game loop, worker lifecycle
    twixt.ts       Game logic (ported from src/twixt.py)
    mcts.ts        MCTS engine
    onnx-player.ts ONNX Runtime wrapper
    worker.ts      Web Worker (runs MCTS + inference off main thread)
    ui.ts          Canvas board renderer + touch/pointer input
    sw.ts          Service Worker (offline caching + COOP/COEP headers)
    naf.ts         Position -> model input tensor encoding
    game-mode.ts   Game mode helpers
docs/              Architecture and research notes
```

### Architecture notes

See [`CLAUDE.md`](CLAUDE.md) for detailed notes on:
- ONNX Runtime WASM binary selection (critical for iOS)
- iOS memory management and the deferred-kill problem
- Service Worker COOP/COEP header injection
- Diagnostic logging system
- AI improvement ideas

---

## License

MIT — see [LICENSE](LICENSE).  
Original twixtbot copyright © 2019 Jordan Lampe.
