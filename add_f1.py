b = "BACKEND/app.py"
s = open(b).read()
if '"prf"' not in s:
    old = '''        out["by_claim"] = [{"domain": d, "attribute": ATTR_LABELS.get(a, a), "accuracy": round(c / n, 4), "pairs": n}
                           for (d, a), (c, n) in sorted(stats.items())]'''
    new = '''        out["by_claim"] = [{"domain": d, "attribute": ATTR_LABELS.get(a, a), "accuracy": round(c / n, 4), "pairs": n}
                           for (d, a), (c, n) in sorted(stats.items())]
        conf = defaultdict(lambda: [0, 0, 0, 0])  # tp, fp, fn, tn per domain
        with open(folder / "test_predictions.csv", newline="") as f:
            for r in csv.DictReader(f):
                y, p = int(r["label"]), int(r["prediction"])
                i = 0 if (y, p) == (1, 1) else 1 if (y, p) == (0, 1) else 2 if (y, p) == (1, 0) else 3
                conf[r["domain"]][i] += 1
                conf["ALL"][i] += 1
        out["prf"] = {}
        for d, (tp, fp, fn, tn) in conf.items():
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            out["prf"][d] = {"precision": round(pr, 4), "recall": round(rc, 4),
                             "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0,
                             "tp": tp, "fp": fp, "fn": fn, "tn": tn}'''
    if s.count(old) != 1:
        raise SystemExit("Could not update BACKEND/app.py - send me a screenshot.")
    open(b, "w").write(s.replace(old, new))

f = "frontend/index.html"
h = open(f).read()
if 'id="r-prf"' not in h:
    old_html = '''        <div class="card"><h2 style="margin:0 0 12px; font-size:17px">Custom model vs CLIP</h2>'''
    new_html = '''        <div class="card"><h2 style="margin:0; font-size:17px">Precision, recall and F1</h2>
          <p class="hint" style="margin:4px 0 12px">Precision: of the claims called consistent, how many were. Recall: of the true claims, how many were accepted. F1 combines both.</p>
          <div class="scroll"><table id="r-prf"></table></div></div>
        <div class="card"><h2 style="margin:0 0 12px; font-size:17px">Confusion matrix (all domains)</h2>
          <div class="scroll"><table id="r-conf"></table></div></div>
        <div class="card"><h2 style="margin:0 0 12px; font-size:17px">Custom model vs CLIP</h2>'''
    if h.count(old_html) != 1:
        raise SystemExit("Could not find the CLIP card in frontend/index.html - send me a screenshot.")
    h = h.replace(old_html, new_html)
    old_js = '''    if (r.clip) {
      $("r-clip").textContent = pct(r.clip.ALL.accuracy);'''
    new_js = '''    if (r.prf) {
      const doms = Object.keys(r.prf).filter((d) => d !== "ALL").sort().concat(["ALL"]);
      $("r-prf").innerHTML = '<tr><th>Domain</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th></tr>' +
        doms.map((d) => `<tr><td>${d === "ALL" ? "<b>All</b>" : (names[d] || d)}</td><td class="num mono">${pct(r.prf[d].precision)}</td><td class="num mono">${pct(r.prf[d].recall)}</td><td class="num mono"><b>${pct(r.prf[d].f1)}</b></td></tr>`).join("");
      const c = r.prf.ALL;
      $("r-conf").innerHTML = '<tr><th></th><th class="num">Predicted consistent</th><th class="num">Predicted inconsistent</th></tr>' +
        `<tr><td>Actually consistent</td><td class="num mono">${c.tp}</td><td class="num mono">${c.fn}</td></tr>` +
        `<tr><td>Actually inconsistent</td><td class="num mono">${c.fp}</td><td class="num mono">${c.tn}</td></tr>`;
    }
    if (r.clip) {
      $("r-clip").textContent = pct(r.clip.ALL.accuracy);'''
    if h.count(old_js) != 1:
        raise SystemExit("Could not find the CLIP loader in frontend/index.html - send me a screenshot.")
    h = h.replace(old_js, new_js)
    open(f, "w").write(h)
print("Precision, recall, F1 and confusion matrix added to the Model results tab.")
