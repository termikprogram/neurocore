# Shared pool — multi-machine job sharing

**Version 0.1.0**

NeuroCore instances running on different machines (and different networks)
find each other, share work, and if one does not answer the **job is taken
over locally**. No inbound port has to be opened.

![Shared pool](../images/pool-diagram.png)

---

## 1 · Why a relay server

```text
local machine : 192.168.1.x/24      (behind the router)
public IP     : <router exit>       (NAT)
```

From outside, `<public IP>:port` lands on the **router** → direct peer-to-peer
does not work. So every node opens a **reverse tunnel** to an SSH server it can
reach:

```text
node A ──ssh -R──▶ server ◀──ssh -R── node B
                     │
      <server>/neurocore/peers.jsonl     (registry, TTL 90 s)
      <server>/neurocore/cluster.token   (cluster token, mode 600)
```

Measured: tunnel setup **6.3 s**, relay job latency **~1.7–2.7 s**.
Because `GatewayPorts` is off the tunnel listens on `127.0.0.1`, so jobs are
carried **through the server as a relay**.

---

## 2 · Configuration (secrets never in code)

Order: **environment variable** → **local settings file**.

### Option A — environment variables

```powershell
setx NC_SERVER  "ubuntu@SERVER-ADDRESS"
setx NC_SSH_KEY "%USERPROFILE%\.ssh\nc_thisMachine"
```

### Option B — local settings file

`.neurocore\peer.json` (**not committed**):

```json
{
  "server": "ubuntu@SERVER-ADDRESS",
  "ssh_key": "C:\\Users\\you\\.ssh\\nc_thisMachine"
}
```

### Cluster token

```text
1. the node connects with its own SSH key
2. <server>/neurocore/cluster.token is read (the first node creates it, mode 600)
3. it is stored locally as .neurocore/peer_token
4. /peer/job is validated against this file (no SSH needed, fast)
5. NC_PEER_TOKEN can pin it manually
```

> The token is **not derived from the SSH key**: every machine uses its own
> key and the secret lives in one place on the server.

---

## 3 · Keys (never shared)

```powershell
# On a new machine — the installer does this for you:
.\kur_istemci.ps1                  # generates a key if missing, prints the command

# Authorization (on a machine that ALREADY has server access, one line):
type "<new-machine>\.pub" | ssh -i "<working key there>" ubuntu@SERVER `
    "cat >> ~/.ssh/authorized_keys"

# Then verify:
.\kur_istemci.ps1 -SadeceAnahtar   # STATUS: working
```

> **The private key never leaves its machine.** Only the public key is passed
> around, and only the person holding server access can authorize it.

---

## 4 · Runtime

```text
start  : reverse tunnel → register → heartbeat (30 s)
stop   : registration removed, tunnel closed
supervision: if the tunnel process dies it is reopened
```

To start with the pool closed: `NC_PEER=0`.

### Job flow

```text
AgentLoop._execute(tool, args)
    │
    ├─ is the tool in ALLOWED_TOOLS? ── no ──▶ LOCAL
    │
    └─ yes ─▶ PeerPool.try_neighbour()
                    ├─ answer ──▶ [remote:<address>]
                    └─ no answer ──▶ LOCAL (work is never lost)
```

Measured against a real server:

```text
web.search  → ran on the neighbour · local registry calls: 0
shell.run   → stayed local (never sent to a peer)
no neighbour → local takeover (2.7 s)
```

---

## 5 · Allowed tools (remote, read-only — 16)

```text
web.search · web.fetch · knowledge.ask · learn.status
code.workspace_map · code.read_file · code.list_directory
code.search_text · code.find_files
doc.kutuphane_ara · doc.api_imzasi_al · doc.surum_karsilastir
doc.godot_api_baglam
asset.katalog · asset.onizleme · asset.anim_onizleme
```

**Forbidden:** `shell.run`, `shell.session_run`, `code.run_command`,
`code.edit_file`, `code.create_file` … Tools that need approval never run
remotely either.

---

## 6 · Endpoints

```text
GET  /peer/info   → node info + allowed tool list (no token required)
POST /peer/job    → {"token","tool","args"} → result
```

- wrong/missing token → **403** `unauthorized: …`
- tool outside the allow-list → `ok=false` `tool cannot run remotely`
- tool that needs approval → `ok=false` `requires approval`

---

## 7 · Verification commands

```powershell
python tools\ag_havuzu.py --liste        # neighbours + liveness
python tools\havuz_uctan_uca.py          # 6 steps: tunnel+register+relay+heartbeat+cleanup
python tools\ajan_is_paylasimi.py        # 5 steps: agent job sharing
```

Expected with two machines:

```text
neighbour count : 1 (excluding self)
  <machine-name>     relay:<port>     age=2.4s   alive=True
```

---

## 8 · Limits (honest)

| Limit | Description |
|---|---|
| Relay latency | ~2 s per job; profitable only for **heavy** jobs |
| Load balancing | none — the first live neighbour is chosen |
| Library content | not shared (capability metadata only) |
| Same user | one user on the server → anyone reaching that user joins the pool |
| Token rotation | delete `cluster.token` on the server; delete `peer_token` on nodes and restart |
