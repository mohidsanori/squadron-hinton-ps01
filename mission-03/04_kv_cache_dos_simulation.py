"""
Mission 03 — Part 4 support: KV-cache Denial-of-Service simulation
Course: Conversational AI (WS 2026/27)
Starship Unit: USSE Noosphere, Squadron Hinton

Numbers and plot for 04_Memory_Hierarchy_and_Safety_Audit.md (Security Sentinel).

Mainframe: one 80 GB H100 serving a LLaMA-3-8B-shaped model in fp16
(16.06 GB weights, 4 GB reserved for activations and the CUDA context -> ~60 GB for KV).

1. Static capacity: how many sessions at N = 65,536 fit before CUDA OOM, for
   MHA, GQA 32:8, MQA and an MLA-style latent cache.
2. Timeline: legitimate crew sessions plus, from turn 10, a rogue client that keeps
   opening long multi-turn sessions and stuffs 8,192 tokens into every turn.
   Three server policies:
     static       - reserve max_len (65,536 tokens) per session up front, no admission check:
                    the first allocation that does not fit is a CUDA OOM and kills the batch
     paged        - PagedAttention: 16-token blocks allocated on demand, admission only if
                    blocks are free; when memory runs out the newest session is preempted
     paged+quota  - paged, plus per-client limits: 2 concurrent sessions, 16,384-token context
   -> 04_dos_memory_timeline.png

Usage:
    python 04_kv_cache_dos_simulation.py
"""

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(HERE, "04_dos_memory_timeline.png")

GB = 1e9
HBM = 80 * GB
WEIGHTS = 16.06 * GB
RESERVED = 4 * GB
KV_BUDGET = HBM - WEIGHTS - RESERVED
LAYERS, HEAD_DIM, Q_HEADS = 32, 128, 32
FP16 = 2
MAX_LEN = 65_536
BLOCK = 16                                   # vLLM default block size (tokens)


def bytes_per_token(kv_heads=None, latent=None):
    """KV bytes per token for the whole model. latent = (d_c, d_rope) for MLA."""
    if latent is not None:
        return (latent[0] + latent[1]) * FP16 * LAYERS
    return 2 * kv_heads * HEAD_DIM * FP16 * LAYERS


ARCHS = [
    ("MHA 32:32", bytes_per_token(32)),
    ("GQA 32:8", bytes_per_token(8)),
    ("MQA 32:1", bytes_per_token(1)),
    ("MLA d_c=512 (+64 RoPE)", bytes_per_token(latent=(512, 64))),
]


def static_capacity():
    print("=== 1. Sessions at N = 65,536 before CUDA OOM (80 GB H100, 8B model, fp16) ===")
    print(f"KV budget: 80 - {WEIGHTS / GB:.2f} (weights) - {RESERVED / GB:.0f} (reserve) = {KV_BUDGET / GB:.2f} GB\n")
    print(f"{'architecture':<24} | {'KiB / token':>11} | {'GB / session':>12} | {'max sessions':>12} | tokens in budget")
    rows = []
    for name, bpt in ARCHS:
        per_session = bpt * MAX_LEN
        n = int(KV_BUDGET // per_session)
        rows.append((name, bpt, per_session, n))
        print(f"{name:<24} | {bpt / 1024:>11.1f} | {per_session / GB:>12.2f} | {n:>12} | {KV_BUDGET / bpt:>15,.0f}")
    print()
    return rows


class Session:
    def __init__(self, sid, client, start, turns, tokens_per_turn):
        self.sid, self.client, self.start = sid, client, start
        self.turns_left, self.tokens_per_turn = turns, tokens_per_turn
        self.tokens = 0
        self.reserved = 0                    # tokens of KV memory held for this session


def simulate(policy, bpt, ticks=60, attack_start=10, attack=True):
    """Returns per-tick memory (GB) by owner, and event counters."""
    capacity_tokens = int(KV_BUDGET // bpt)
    sessions, next_id = [], 0
    stats = dict(oom=0, first_oom=None, legit_requested=0, legit_served=0, rejected=0, preempted=0,
                 legit_preempted=0)
    timeline = []

    def reserved_total():
        return sum(s.reserved for s in sessions)

    for t in range(ticks):
        # arrivals: 1 crew session every 2 turns (20 turns x 350 tokens); attacker 4 per turn (forever x 8192)
        arrivals = [Session(None, "crew", t, 20, 350)] if t % 2 == 0 and t <= ticks - 20 else []
        stats["legit_requested"] += 20 * len(arrivals)
        if attack and t >= attack_start:
            arrivals += [Session(None, "rogue", t, 10 ** 9, 8192) for _ in range(4)]
        for s in arrivals:
            s.sid, next_id = next_id, next_id + 1
            if policy == "paged+quota" and s.client == "rogue":
                if sum(1 for x in sessions if x.client == "rogue") >= 2:
                    stats["rejected"] += 1
                    continue
            if policy == "static":
                s.reserved = MAX_LEN
                if reserved_total() + s.reserved > capacity_tokens:
                    stats["oom"] += 1                         # allocation fails: CUDA OOM, batch is lost
                    stats["first_oom"] = stats["first_oom"] if stats["first_oom"] is not None else t
                    stats["legit_preempted"] += sum(1 for x in sessions if x.client == "crew")
                    sessions = []
                    continue
            elif reserved_total() + BLOCK > capacity_tokens:
                stats["rejected"] += 1
                continue
            sessions.append(s)

        # one turn for every live session
        limit = 16_384 if policy == "paged+quota" else MAX_LEN
        for s in list(sessions):
            if s not in sessions:
                continue
            add = min(s.tokens_per_turn, limit - s.tokens)
            if add <= 0:                                       # context limit reached: request refused
                if policy == "paged+quota":
                    stats["rejected"] += 1
                s.turns_left = 0
                continue
            if policy != "static":
                need = math.ceil((s.tokens + add) / BLOCK) * BLOCK - s.reserved
                while reserved_total() + need > capacity_tokens and len(sessions) > 1:
                    victim = max(sessions, key=lambda x: x.start * 10 ** 6 + x.sid)   # newest first
                    sessions.remove(victim)
                    stats["preempted"] += 1
                    stats["legit_preempted"] += victim.client == "crew"
                    if victim is s:
                        break
                if s not in sessions:
                    continue
                s.reserved += need
            s.tokens += add
            s.turns_left -= 1
            if s.client == "crew":
                stats["legit_served"] += 1
        sessions = [s for s in sessions if s.turns_left > 0]

        used = {c: sum(s.tokens for s in sessions if s.client == c) * bpt / GB for c in ("crew", "rogue")}
        waste = (reserved_total() - sum(s.tokens for s in sessions)) * bpt / GB
        stats["peak_crew"] = max(stats.get("peak_crew", 0), sum(1 for s in sessions if s.client == "crew"))
        timeline.append(dict(t=t, crew=used["crew"], rogue=used["rogue"], waste=waste))
    stats["served_pct"] = 100 * stats["legit_served"] / max(stats["legit_requested"], 1)
    return timeline, stats


def run_timeline(arch="GQA 32:8"):
    bpt = dict(ARCHS)[arch]
    print(f"=== 2. 60-turn attack timeline, {arch}, rogue client from turn 10 ===")
    print("Crew: 1 new session every 2 turns, 20 turns x 350 tokens each (~10 live sessions).")
    print("Rogue: from turn 10, 4 new sessions per turn, 8,192 tokens per turn, never closes.\n")
    print(f"{'policy':<12} | {'attack':>6} | {'OOM panics':>10} | {'first OOM':>9} | {'crew turns served':>17} | "
          f"{'crew sessions lost':>18} | {'rejected req.':>13} | {'peak empty reserve':>18}")
    results = {}
    for policy in ("static", "paged", "paged+quota"):
        for attack in (False, True):
            tl, st = simulate(policy, bpt, attack=attack)
            if attack:
                results[policy] = (tl, st)
            waste = max(r["waste"] for r in tl)
            waste_s = f"{waste:.1f} GB" if waste >= 0.1 else f"{waste * 1000:.1f} MB"
            print(f"{policy:<12} | {'yes' if attack else 'no':>6} | {st['oom']:>10} | {str(st['first_oom']):>9} | "
                  f"{st['served_pct']:>16.0f}% | {st['legit_preempted']:>18} | {st['rejected']:>13} | {waste_s:>18}")
    print("(lost = killed by an OOM panic, or preempted. vLLM would swap out or recompute a preempted")
    print(" sequence later, but while the attack lasts there is no memory to bring it back into.)\n")
    return results


def plot(results, path=PLOT_PATH):
    from matplotlib.figure import Figure  # no pyplot: leaves a notebook's backend alone

    fig = Figure(figsize=(16, 4.4))
    axes = fig.subplots(1, 3, sharey=True)
    for ax, (policy, (tl, st)) in zip(axes, results.items()):
        t = [r["t"] for r in tl]
        ax.stackplot(t, [r["crew"] for r in tl], [r["rogue"] for r in tl], [r["waste"] for r in tl],
                     labels=["crew KV (used)", "rogue KV (used)", "reserved but empty"],
                     colors=["#2ca02c", "#d62728", "#bbbbbb"])
        ax.axhline(KV_BUDGET / GB, color="black", ls="--", lw=1)
        ax.axvline(10, color="#d62728", ls=":", lw=1)
        ax.set_title(f"{policy}: {st['oom']} OOM, crew served {st['served_pct']:.0f}%")
        ax.set_xlabel("turn")
    axes[0].set_ylabel("KV-cache memory (GB)")
    axes[0].text(0.5, KV_BUDGET / GB + 1, "KV budget 60 GB", fontsize=8)
    axes[0].legend(loc="center left", fontsize=8)
    fig.suptitle("KV-cache DoS on one 80 GB H100 (GQA 32:8, 8B model); rogue client starts at turn 10", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[SAVED] {os.path.basename(path)}")


if __name__ == "__main__":
    static_capacity()
    plot(run_timeline())
