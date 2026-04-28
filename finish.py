#!/usr/bin/env python3
"""Finish remaining 9 papers."""
import json, os, re, subprocess, sys, time, requests
from pathlib import Path

REPO = Path("/home/vincent/post-transformer-research")
SKILL = (Path("/home/vincent/.openclaw/workspace/skills/read-paper/SKILL.md")).read_text()
with open("/home/vincent/.openclaw/agents/main/agent/auth-profiles.json") as f:
    KEY = json.load(f)["profiles"]["deepseek:default"]["key"]
LOG = "/tmp/batch-finish.log"

def log(m): 
    with open(LOG, "a") as f: f.write(f"[{time.strftime('%H:%M:%S')}] {m}\n")

papers = sorted(REPO.glob("papers/*.md"), key=lambda p: len(p.read_text().splitlines()))
done = 0
for p in papers:
    content = p.read_text()
    lines = len(content.splitlines())
    url = re.search(r'https?://arxiv\.org/abs/[0-9]+\.[0-9]+', content)
    if lines >= 60 or "Paper reading generated" in content or not url:
        continue
    
    url = url.group()
    title = content.split("\n")[0].strip()
    log(f"PROC {p.name}: {title[:50]}...")
    
    r = requests.post("https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {KEY}"},
        json={"model": "deepseek-chat", "messages": [
            {"role": "system", "content": "严格按照 skill 生成中文论文精读。直接输出。"},
            {"role": "user", "content": f"论文：{title}\n链接：{url}\n文件：{p.name}\n输出路径：{p}\n要求：保留抬头行，紧跟写入精读（中文）。按 skill 规范包含动机链、架构、公式、实验、代码、位置。skill：{SKILL}"}
        ], "max_tokens": 64000, "temperature": 0.7}, timeout=300)
    
    result = r.json()["choices"][0]["message"]["content"]
    # Strip leading meta-commentary, keep from first # header
    lines_r = result.split("\n")
    clean = []
    found = False
    for l in lines_r:
        if l.startswith("# ") and not found: found = True
        if found: clean.append(l)
    if not clean: clean = lines_r
    
    p.write_text("\n".join(clean))
    log(f"WRITE {p.name}: {len(clean)} lines")
    done += 1
    subprocess.run(["git", "-C", str(REPO), "add", f"papers/{p.name}"], capture_output=True)
    subprocess.run(["git", "-C", str(REPO), "commit", "-m", f"deep: {title[:50]}"], capture_output=True, timeout=30)
    time.sleep(5)

subprocess.run(["git", "-C", str(REPO), "push", "origin", "main"], capture_output=True, timeout=30)
log(f"DONE. Processed: {done}")
