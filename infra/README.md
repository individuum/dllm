# `infra/` — Phase 1 VPS deploy

Files used to host `dllm-coord` on a single VPS behind nginx + Let's Encrypt.

| File | Destination |
|---|---|
| `dllm-coord.service` | `/etc/systemd/system/dllm-coord.service` |
| `nginx-dllm.conf`    | `/etc/nginx/sites-available/dllm` (symlink into `sites-enabled/`) |

## First-time deploy (Debian 13, rootful Podman, nginx, certbot installed)

```bash
# upload code
ssh root@<host> "mkdir -p /opt/dllm"
rsync -az --delete --exclude .venv --exclude data/cache --exclude coord/state \
      --exclude tests --exclude .pytest_cache --exclude '*.egg-info' \
      ./ root@<host>:/opt/dllm/

ssh root@<host> bash <<'EOF'
set -euxo pipefail
cd /opt/dllm
podman build -t localhost/dllm:local .
podman volume inspect dllm-data >/dev/null 2>&1 || podman volume create dllm-data

install -m 0644 infra/dllm-coord.service /etc/systemd/system/dllm-coord.service
install -m 0644 infra/nginx-dllm.conf    /etc/nginx/sites-available/dllm
ln -sf ../sites-available/dllm /etc/nginx/sites-enabled/dllm
nginx -t
systemctl reload nginx

certbot --nginx -d dllm.planetbass.de --non-interactive --agree-tos --email nick@planetbass.de --redirect

systemctl daemon-reload
systemctl enable --now dllm-coord.service
EOF

curl -fsS https://dllm.planetbass.de/health
```

## Iterating

```bash
rsync ... ./ root@<host>:/opt/dllm/
ssh root@<host> "cd /opt/dllm && podman build -t localhost/dllm:local . && systemctl restart dllm-coord"
```

Coordinator state lives in the `dllm-data` podman volume — survives container rebuilds. Wipe to reset:
```bash
ssh root@<host> "systemctl stop dllm-coord && podman volume rm dllm-data && podman volume create dllm-data && systemctl start dllm-coord"
```
