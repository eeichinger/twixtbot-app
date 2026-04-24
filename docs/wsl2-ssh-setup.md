# WSL2 SSH Setup — Remote Access from Mac to Windows PC

Connect from your M1 MacBook to your Windows PC's WSL2 environment
for remote PyTorch training, Claude Code Remote Control, and general
development.

## Why

- **Claude Code runs better on Mac** (native terminal, no WSL2 overhead)
- **GPU training needs the PC** (RTX 5070 Ti)
- SSH gives you direct terminal access alongside Claude Code Remote Control

## Architecture

```
Mac (SSH client)
  |
  |  port 2222/tcp over LAN
  v
Windows host (netsh port proxy)
  |
  |  forwards to WSL2 VM IP:2222
  v
WSL2 Ubuntu (sshd on port 2222)
  |
  |  nvidia-smi / PyTorch / CUDA
  v
RTX 5070 Ti
```

Port 2222 is used instead of 22 to avoid conflicts with Windows' own
OpenSSH server (if installed).

---

## Step 1 — Generate an SSH key on your Mac (if you don't have one)

```bash
# On Mac
ssh-keygen -t ed25519 -C "erich@macbook"
# Accept defaults, set a passphrase if you like

cat ~/.ssh/id_ed25519.pub
# Copy this output — you'll paste it in Step 2
```

---

## Step 2 — Run the WSL2 setup script

In your WSL2 terminal:

```bash
cd /root/workspaces/twixtbot-app
sudo bash scripts/wsl2-sshd-setup.sh
```

This installs openssh-server, configures it on port 2222 with pubkey-only
auth, and prompts you to paste your Mac's public key.

If you skipped the key paste, add it manually:

```bash
echo 'ssh-ed25519 AAAA... erich@macbook' >> ~/.ssh/authorized_keys
```

### Verify sshd is running

```bash
ss -tlnp | grep 2222
# Should show sshd listening
```

---

## Step 3 — Set up Windows port forwarding

WSL2 runs behind a NAT — your Mac can't reach the WSL2 IP directly.
Port forwarding on the Windows host bridges the gap.

Open **PowerShell as Administrator** and run:

```powershell
cd C:\path\to\twixtbot-app
.\scripts\wsl2-port-forward.ps1
```

This script:
1. Gets the current WSL2 IP
2. Creates a `netsh portproxy` rule: `Windows:2222 -> WSL2:2222`
3. Adds a Windows Firewall rule allowing inbound TCP on 2222 (Private network only)
4. Creates a scheduled task to refresh the forwarding at logon (WSL2 IP changes on reboot)

### Verify port forwarding

```powershell
netsh interface portproxy show v4tov4
# Should show 0.0.0.0:2222 -> <wsl2-ip>:2222
```

---

## Step 4 — Configure SSH on your Mac

Find your Windows PC's LAN IP:

```powershell
# On Windows (PowerShell)
ipconfig | findstr "IPv4"
# e.g., 192.168.1.42
```

Add this to `~/.ssh/config` on your Mac:

```
Host pc
    HostName 192.168.1.42
    Port 2222
    User root
    # Optional: forward a port for Vite dev server
    # LocalForward 5173 localhost:5173
```

Replace `192.168.1.42` with your PC's actual LAN IP. If your router
supports it, assign a static DHCP lease so this IP doesn't change.

---

## Step 5 — Test the connection

```bash
# On Mac
ssh pc
```

You should land in your WSL2 shell without a password prompt.

### Verify GPU access

```bash
ssh pc "nvidia-smi"
# Should show your RTX 5070 Ti

ssh pc "cd /root/workspaces/twixtbot-app && python -c 'import torch; print(torch.cuda.get_device_name())'"
# Should print: NVIDIA GeForce RTX 5070 Ti
```

---

## Usage Patterns

### Quick: kick off training from Mac

```bash
ssh pc "cd /root/workspaces/twixtbot-app && python train_loop.py"
```

### Interactive: full terminal session

```bash
ssh pc
cd /root/workspaces/twixtbot-app
python train_loop.py
# Ctrl+C to stop, Ctrl+D to disconnect
```

### Long-running: tmux/screen for persistent sessions

```bash
ssh pc
tmux new -s train
cd /root/workspaces/twixtbot-app
python train_loop.py
# Ctrl+B, D to detach — training continues

# Reconnect later:
ssh pc
tmux attach -t train
```

### Sync code from Mac to PC

```bash
# On Mac — push code changes to PC
rsync -avz --exclude='models/*.pt' --exclude='node_modules/' \
  ~/workspaces/twixtbot-app/ pc:/root/workspaces/twixtbot-app/

# On Mac — pull trained model back
rsync pc:/root/workspaces/twixtbot-app/models/v1.pt ./models/
```

Or use git push/pull if you prefer (see CLAUDE.md git workflow).

### Claude Code Remote Control (recommended)

For the best experience, combine SSH with Claude Code Remote Control:

```bash
# On PC (via SSH or directly)
cd /root/workspaces/twixtbot-app
claude remote-control --name "twixtbot training"

# Opens a URL — access it from your Mac's browser at claude.ai/code
# Full Claude Code experience, running on the PC with GPU access
```

---

## Troubleshooting

### "Connection refused" from Mac

1. **Is sshd running in WSL2?**
   ```bash
   # In WSL2
   ss -tlnp | grep 2222
   ```
   If not: `sudo service ssh restart`

2. **Is port forwarding active on Windows?**
   ```powershell
   netsh interface portproxy show v4tov4
   ```
   If empty or wrong IP: re-run `.\scripts\wsl2-port-forward.ps1`

3. **Is the firewall allowing it?**
   ```powershell
   Get-NetFirewallRule -DisplayName "WSL2 SSH*" | Format-Table
   ```

4. **Is WSL2 even running?**
   ```powershell
   wsl --list --running
   ```

### "Permission denied (publickey)"

Your Mac's public key isn't in WSL2's `~/.ssh/authorized_keys`:

```bash
# On Mac — copy your key to WSL2
ssh-copy-id -p 2222 root@192.168.1.42

# Or manually:
cat ~/.ssh/id_ed25519.pub | ssh pc "cat >> ~/.ssh/authorized_keys"
```

### Port forwarding breaks after reboot

WSL2 gets a new IP on every Windows restart. The scheduled task created
by `wsl2-port-forward.ps1` should handle this automatically. If it
doesn't fire:

```powershell
# Manual refresh (Admin PowerShell)
.\scripts\wsl2-port-forward-refresh.ps1

# Or re-run the full setup
.\scripts\wsl2-port-forward.ps1
```

### sshd doesn't start automatically in WSL2

WSL2 without systemd doesn't auto-start services. Add to your
`~/.bashrc` or create `/etc/wsl.conf`:

```ini
# /etc/wsl.conf
[boot]
command = service ssh start
```

This requires WSL2 version 0.67.6+ (Windows 11 22H2+). Verify with:

```powershell
wsl --version
```

### Slow SSH connection (hangs for seconds before prompt)

Usually DNS reverse lookup. The setup script disables this (`UseDNS no`),
but if you're still seeing delays:

```bash
# In WSL2
echo "UseDNS no" | sudo tee -a /etc/ssh/sshd_config.d/wsl2.conf
sudo service ssh restart
```

---

## Security Notes

- **Private network only**: The firewall rule allows connections only on
  the Private network profile. Public/Guest networks are blocked.
- **No password auth**: Only pubkey authentication is allowed. No risk
  of brute-force attacks.
- **Port 2222**: Non-standard port reduces drive-by scanning noise.
- **LAN-only**: Your router's NAT prevents external access unless you
  explicitly set up port forwarding on the router (don't do this).
