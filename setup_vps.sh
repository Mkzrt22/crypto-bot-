#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# VPS setup script for AI Crypto Trader
# Tested on Ubuntu 22.04 / 24.04
#
# Usage:
#   chmod +x setup_vps.sh
#   ./setup_vps.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=${PYTHON:-python3}
VENV_DIR="$INSTALL_DIR/venv"
SERVICE_FILE="/etc/systemd/system/crypto-trader.service"
CURRENT_USER=$(whoami)

echo "============================================"
echo "  AI Crypto Trader — VPS Setup"
echo "  Install dir : $INSTALL_DIR"
echo "  User        : $CURRENT_USER"
echo "============================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv nginx ufw curl

# ── 2. Python venv + dependencies ────────────────────────────────────────────
echo ""
echo "[2/6] Creating Python venv and installing requirements..."
$PYTHON -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$INSTALL_DIR/requirements.txt" -q
echo "  Done."

# ── 3. .env file ─────────────────────────────────────────────────────────────
echo ""
echo "[3/6] Setting up .env config..."
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    echo "  Created .env from .env.example"
    echo "  >>> Edit $INSTALL_DIR/.env before starting the bot <<<"
else
    echo "  .env already exists — skipping."
fi

# ── 4. Firewall ───────────────────────────────────────────────────────────────
echo ""
echo "[4/6] Configuring firewall (ufw)..."
sudo ufw allow OpenSSH   >/dev/null
sudo ufw allow 80/tcp    >/dev/null   # HTTP  (nginx)
sudo ufw allow 443/tcp   >/dev/null   # HTTPS (nginx + certbot)
# Port 8501 is NOT opened publicly — nginx proxies it from localhost
sudo ufw --force enable  >/dev/null
echo "  Firewall: SSH(22), HTTP(80), HTTPS(443) allowed."
echo "  Port 8501 is kept internal (nginx proxies it)."

# ── 5. Nginx config ───────────────────────────────────────────────────────────
echo ""
echo "[5/6] Installing nginx reverse proxy config..."
NGINX_CONF="/etc/nginx/sites-available/crypto-trader"
if [ ! -f "$NGINX_CONF" ]; then
    sudo cp "$INSTALL_DIR/nginx-crypto-trader.conf" "$NGINX_CONF"
    sudo ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/crypto-trader
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t && sudo systemctl reload nginx
    echo "  Nginx configured. Update server_name in $NGINX_CONF then reload nginx."
else
    echo "  Nginx config already exists — skipping."
fi

# ── 6. systemd service ───────────────────────────────────────────────────────
echo ""
echo "[6/6] Installing systemd service..."
# Patch the service file with actual user + paths
sed \
    -e "s|User=ubuntu|User=$CURRENT_USER|g" \
    -e "s|WorkingDirectory=.*|WorkingDirectory=$INSTALL_DIR|g" \
    -e "s|ExecStart=.*|ExecStart=$VENV_DIR/bin/python $INSTALL_DIR/main.py start|g" \
    "$INSTALL_DIR/crypto-trader.service" \
    | sudo tee "$SERVICE_FILE" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable crypto-trader
echo "  Service installed and enabled."

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Edit your config:"
echo "     nano $INSTALL_DIR/.env"
echo ""
echo "  2. (Optional) Point a domain at this VPS IP, then:"
echo "     sudo nano $NGINX_CONF           # set server_name"
echo "     sudo systemctl reload nginx"
echo "     sudo apt install certbot python3-certbot-nginx -y"
echo "     sudo certbot --nginx -d yourdomain.com"
echo ""
echo "  3. Start the bot:"
echo "     sudo systemctl start crypto-trader"
echo ""
echo "  4. Check logs:"
echo "     sudo journalctl -u crypto-trader -f"
echo ""
echo "  5. Open dashboard on your phone:"
echo "     http://$(curl -s ifconfig.me 2>/dev/null || echo YOUR_VPS_IP)"
echo "     (or https://yourdomain.com if you set up SSL)"
echo ""
