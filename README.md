# TwixT vs AI

A free web app that lets you play the board game **TwixT** against a computer
opponent — or against a friend on the same device — with no download or account
required.

**Play now:** https://eeichinger.github.io/twixtbot-app/

The computer opponent is very strong (it has beaten top-rated players on
Little Golem). You can tune its think time to make it as tough or as easy as
you like. The app also includes tools to help you understand the game, learn
from your mistakes, and explore thousands of real expert games.

---

## How to Play

### Opening the app

Go to https://eeichinger.github.io/twixtbot-app/ in your browser. Everything
loads in a few seconds. After the first visit it works entirely offline.

### Install to your home screen (optional)

You can add the app to your home screen for one-tap access — no App Store needed.

- **iPhone/iPad:** tap the Share button → "Add to Home Screen"
- **Android:** tap the three-dot menu → "Add to Home Screen" or "Install app"
- **Desktop Chrome/Edge:** click the install icon in the address bar

Once installed it launches and behaves exactly like a native app, and still
works offline.

### Choose a mode

| Mode | What happens |
|------|-------------|
| **vs Computer** | You play as Blue; the computer plays as Orange |
| **vs Player** | Two people take turns on the same screen |

Tap **Start Game** to begin.

### Placing a peg

- **Tap or click** any empty dot on the board to place your peg there
- On a touchscreen, **press and drag** your finger to preview where the peg will
  go, then lift your finger to confirm. The board label (e.g. "h5") appears next
  to the peg so you always know the exact position before you commit.
- On desktop, **hover** over the board to see the cell label, then click to place.
- Bridges between your pegs are drawn automatically whenever they don't cross an
  existing bridge.

### Winning

Connect your two border edges with an unbroken chain of pegs and bridges.
Blue connects left ↔ right; Orange connects top ↔ bottom. The winning chain is
highlighted in purple when the game ends.

---

## Brief Rules

TwixT is played on a 24×24 grid of dots.

- Players take turns placing one peg per turn on any empty dot
- After placing, the game automatically adds bridges to nearby pegs that are
  a chess knight's move away — but only if no existing bridge would cross the new one
- **Bridges can never cross.** This is the heart of the game's tactics: blocking
  your opponent's connections while building your own
- **Swap rule:** the second player (Orange) may "swap" — instead of placing a new
  peg they take the first peg for themselves and switch sides. This one-time option
  keeps the game fair no matter who goes first.
- The first player to connect their two edges wins

Full rules: https://en.wikipedia.org/wiki/TwixT

---

## Features

### Playing the game

- **Computer opponent** — a strong AI that thinks ahead. Set its think time from
  5 to 60 seconds: shorter = faster and easier, longer = stronger. Your choice is
  remembered between sessions.
- **vs Player mode** — two people take turns on the same screen, great for playing
  side by side or teaching someone the game
- **Touch drag-to-preview** — press and drag on touchscreens to see exactly where
  your peg will land before you release. A label shows the board coordinate so
  there are no surprises.
- **Hover tooltip** — on desktop, hovering over the board shows the cell label so
  you can plan your move precisely before clicking
- **Swap rule** — the app handles the swap option automatically at the right moment.
  In vs-Computer mode the AI decides whether to swap intelligently, so you can see
  what a good player would do.

### Reviewing and correcting your moves

- **Undo** — take back your last move. In vs-Computer mode it undoes both your move
  and the computer's reply together, so it's always your turn again afterwards.
- **Redo** — put the moves back if you change your mind after undoing
- **Resign** — concede the game cleanly rather than playing to the bitter end
- **Save** — download the current game as a `.tsgf` file so you can share it or
  review it later

### Learning tools

These features help you understand what is happening on the board, see where you
went wrong, and get better over time.

- **Hint** — tap "Hint" and the computer plays your move for you. Useful when
  you're stuck or want to see what a strong player would do in a position.

- **Win probability bar** — after each computer move, a coloured bar in the header
  shows who is ahead: tilting blue means you're winning, tilting orange means the
  computer is ahead. Watch it shift as the game develops.

- **Top-move panel** — after each computer move, a collapsible panel shows the
  three moves the computer rated highest, together with how strongly it preferred
  each one. Compare them with what you played to spot better alternatives.

- **Evaluation chart** — a small line graph tracks the balance of the game from
  move to move. You can see at a glance which single move changed things the most.

- **Policy heatmap** — tap the "Heatmap" button to overlay the board with colours
  showing where the best moves are right now:
  - **Green** — the computer's top choices
  - **Cyan** — decent options worth considering
  - **Blue** — less important squares

  Great for beginners who aren't sure where to look on the board. Available in
  vs-Computer mode, vs-Player mode, and during game replays.

### Exploring real games

The app connects to [Little Golem](https://www.littlegolem.net/), the world's
largest TwixT community, so you can study expert games without leaving the app.

- **Search by player name** — type any Little Golem player's name to browse all
  their finished TwixT games
- **Search by game ID** — go straight to any specific game if you know its number
- **Filter by result** — show only wins, only losses, only draws, or all games
- **Filter by opponent** — narrow the list to games against a specific player
  (e.g. "show only my losses against TwixtBot")
- **Replay any game** — step through a game one move at a time with the ← →
  buttons, or use the left/right arrow keys on a keyboard
- **Analyse any position** — while replaying, tap "Analyse" to get the computer's
  opinion on the current position: win probability and the three best moves shown
  instantly
- **Heatmap during replay** — tap "Heatmap" at any point in a replay to see which
  squares were most promising at that exact moment in the game
- **Full game analysis** — tap "Analyse game" to score every move in the game
  automatically. Each move in the list is colour-coded:
  - **Green** — a great move (the computer's top choice or very close to it)
  - **Yellow** — a reasonable move, but a better one was available
  - **Red** — a significant mistake that hurt the position

  A chart below shows how the game balance shifted with every move, making it
  easy to find the key turning point.
- **Paste or upload a game file** — got a `.tsgf` file? Paste its text or upload
  it to replay it immediately

### Platform

- **Works offline** — after your first visit the app runs entirely on your device
  with no internet needed. The computer opponent, all analysis tools, and game
  replays all work offline (browsing Little Golem requires a connection).
- **No account or installation required** — just open the link and play
- **Automatic updates** — when a new version is ready, it installs quietly in the
  background while you're on the start screen

---

## Attribution

This project builds on [**twixtbot**](https://github.com/BonyJordan/twixtbot) by
**Jordan Lampe** (MIT License, 2019) — a command-line AI that plays TwixT using a
self-trained neural network. The original game logic and model weights are Jordan's
work.

The web app — TypeScript port, AI engine, offline PWA, board UI, touch controls,
iOS memory optimisations, Little Golem integration, and all the analysis and
learning features — was designed and built by
[**Claude**](https://claude.ai) (Anthropic's AI assistant).

---

## Developer notes

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

### Tests

```bash
cd webapp
npm test -- --run
```

### Project structure

```
src/               Original Python/TensorFlow bot (Jordan Lampe)
models/            Trained model weights
tools/             Model export and quantization scripts
webapp/
  src/
    main.ts        App shell, game loop, UI wiring
    twixt.ts       Game logic
    mcts.ts        Monte Carlo Tree Search engine
    onnx-player.ts Neural network inference wrapper
    worker.ts      Web Worker — runs AI off the main thread
    ui.ts          Canvas board renderer and touch/pointer input
    sw.ts          Service Worker — offline caching
    naf.ts         Position encoding for the neural network
    lg-api.ts      Little Golem API client
    lg-sgf.ts      TSGF game file parser
docs/
  planned-features.md   Feature backlog and status
  CLAUDE.md             Architecture decisions and iOS notes
```

See [`CLAUDE.md`](CLAUDE.md) for detailed technical notes on iOS memory
management, Service Worker setup, and AI architecture.

---

## License

MIT — see [LICENSE](LICENSE).  
Original twixtbot copyright © 2019 Jordan Lampe.
