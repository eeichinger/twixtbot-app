# wsl2-port-forward.ps1 — Set up Windows port forwarding to WSL2 sshd
#
# WHY: WSL2 runs behind a NAT. Your Mac can reach the Windows host IP,
#       but not the WSL2 VM IP directly. This script forwards
#       Windows:2222 -> WSL2:2222 so your Mac can SSH in.
#
# Usage: Run in an ELEVATED PowerShell (Run as Administrator)
#
#   .\scripts\wsl2-port-forward.ps1
#
# This script:
#   1. Gets the current WSL2 IP
#   2. Sets up netsh port proxy (Windows:2222 -> WSL2:2222)
#   3. Adds a firewall rule to allow inbound connections on port 2222
#   4. Creates a scheduled task to re-apply on boot (WSL2 IP changes)

$ErrorActionPreference = "Stop"
$SSH_PORT = 2222

Write-Host "=== WSL2 Port Forwarding Setup ===" -ForegroundColor Cyan
Write-Host ""

# --- Check admin privileges ---------------------------------------------------

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: Run this script as Administrator." -ForegroundColor Red
    Write-Host "  Right-click PowerShell -> Run as Administrator"
    exit 1
}

# --- Get WSL2 IP --------------------------------------------------------------

Write-Host "[1/4] Getting WSL2 IP..."
$wslIp = (wsl hostname -I).Trim().Split(" ")[0]
if (-not $wslIp) {
    Write-Host "  ERROR: Could not get WSL2 IP. Is WSL2 running?" -ForegroundColor Red
    exit 1
}
Write-Host "  WSL2 IP: $wslIp"

# --- Set up port proxy --------------------------------------------------------

Write-Host "[2/4] Setting up port forwarding (Windows:$SSH_PORT -> WSL2:$SSH_PORT)..."

# Remove existing rule (ignore errors if none exists)
netsh interface portproxy delete v4tov4 listenport=$SSH_PORT listenaddress=0.0.0.0 2>$null

# Add the new rule
netsh interface portproxy add v4tov4 `
    listenport=$SSH_PORT listenaddress=0.0.0.0 `
    connectport=$SSH_PORT connectaddress=$wslIp

Write-Host "  Done. Current port proxy rules:"
netsh interface portproxy show v4tov4

# --- Firewall rule ------------------------------------------------------------

Write-Host "[3/4] Adding firewall rule..."

$ruleName = "WSL2 SSH (port $SSH_PORT)"
$existingRule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue

if ($existingRule) {
    Write-Host "  Firewall rule already exists, updating..."
    Remove-NetFirewallRule -DisplayName $ruleName
}

New-NetFirewallRule `
    -DisplayName $ruleName `
    -Direction Inbound `
    -LocalPort $SSH_PORT `
    -Protocol TCP `
    -Action Allow `
    -Profile Private | Out-Null

Write-Host "  Done. Rule '$ruleName' allows inbound TCP on port $SSH_PORT (Private profile)."

# --- Scheduled task to refresh on boot ----------------------------------------

Write-Host "[4/4] Creating scheduled task to refresh port forwarding on boot..."

$taskName = "WSL2 SSH Port Forward"
$scriptBlock = @"
`$wslIp = (wsl hostname -I).Trim().Split(' ')[0]
if (`$wslIp) {
    netsh interface portproxy delete v4tov4 listenport=$SSH_PORT listenaddress=0.0.0.0 2>`$null
    netsh interface portproxy add v4tov4 listenport=$SSH_PORT listenaddress=0.0.0.0 connectport=$SSH_PORT connectaddress=`$wslIp
}
"@

# Write the refresh script next to this script
$refreshScript = Join-Path $PSScriptRoot "wsl2-port-forward-refresh.ps1"
Set-Content -Path $refreshScript -Value $scriptBlock
Write-Host "  Wrote refresh script to $refreshScript"

# Remove existing task if present
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$refreshScript`""

$trigger = New-ScheduledTaskTrigger -AtLogon
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -RunLevel Highest `
    -Description "Refresh WSL2 port forwarding after boot (WSL2 IP changes each restart)" | Out-Null

Write-Host "  Done. Task '$taskName' will run at logon."

# --- Summary ------------------------------------------------------------------

# Get the Windows LAN IP for the user
$winIp = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notmatch "Loopback|vEthernet" -and $_.PrefixOrigin -eq "Dhcp" } | Select-Object -First 1).IPAddress
if (-not $winIp) {
    $winIp = "<your-windows-lan-ip>"
}

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "From your Mac, you can now SSH in with:"
Write-Host "  ssh -p $SSH_PORT <user>@$winIp" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or add this to your Mac's ~/.ssh/config (see docs/wsl2-ssh-setup.md)"
Write-Host ""
