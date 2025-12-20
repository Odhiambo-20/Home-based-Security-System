#!/bin/bash
# Deployment script for Biometric Security System

set -e

if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (sudo)"
    exit 1
fi

echo "Deploying Biometric Security System..."

# Install binary
install -m 755 build/biometric_security /usr/local/bin/

# Install configuration
mkdir -p /etc/biometric_security
cp -r config/* /etc/biometric_security/

# Install models
mkdir -p /usr/local/share/biometric_security/models
cp -r models/* /usr/local/share/biometric_security/models/

# Create log directory
mkdir -p /var/log/biometric_security
chmod 750 /var/log/biometric_security

# Install systemd service
cat > /etc/systemd/system/biometric-security.service << 'SERVICE'
[Unit]
Description=Biometric Security System
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/biometric_security
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
SERVICE

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable biometric-security.service

echo "Deployment complete!"
echo "Start service: sudo systemctl start biometric-security"
