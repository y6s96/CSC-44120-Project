b = "BACKEND/app.py"
s = open(b).read()
if '"history"' not in s:
    old = '''                         "by_domain": m.get("test_accuracy_by_domain", {})}'''
    new = '''                         "by_domain": m.get("test_accuracy_by_domain", {}),
                         "history": [{"epoch": h["epoch"], "train_acc": h["train_acc"], "val_acc": h["val_acc"]}
                                     for h in m.get("history", [])],
                         "pick_one": m.get("pick_one_accuracy", {}), "backbone": m.get("backbone", "small"),
                         "image_size": m.get("image_size", 128)}'''
    if s.count(old) != 1:
        raise SystemExit("Could not update BACKEND/app.py - send me a screenshot.")
    s = s.replace(old, new)
    open(b, "w").write(s)

f = "frontend/index.html"
h = open(f).read()
if 'id="r-curve"' not in h:
    old_html = '''        <div class="card"><h2 style="margin:0 0 12px; font-size:17px">By type of claim</h2><div class="scroll"><table id="r-claims"></table></div></div>'''
    new_html = '''        <div class="card"><h2 style="margin:0; font-size:17px">Learning curve</h2>
          <p class="hint" style="margin:4px 0 10px" id="r-curve-note">Accuracy on the training photos (grey) and on unseen validation photos (blue), after each pass through the data.</p>
          <svg id="r-curve" viewBox="0 0 640 260" style="width:100%; height:auto; display:block"></svg></div>
        <div class="card"><h2 style="margin:0; font-size:17px">Picking the right claim out of all options</h2>
          <p class="hint" style="margin:4px 0 14px">Harder than yes/no: the true claim must beat every other claim of its kind. The dark mark is the chance level (1 / number of options).</p>
          <div class="bars" id="r-pick"></div></div>
        <div class="card"><h2 style="margin:0 0 12px; font-size:17px">By type of claim</h2><div class="scroll"><table id="r-claims"></table></div></div>'''
    if h.count(old_html) != 1:
        raise SystemExit("Could not find the results card in frontend/index.html - send me a screenshot.")
    h = h.replace(old_html, new_html)

    old_js = '''    const head = '<tr><th>Domain</th><th>Claim</th><th class="num">Accuracy</th><th class="num">Pairs</th></tr>';'''
    new_js = '''    if (r.custom && r.custom.history && r.custom.history.length) drawCurve(r.custom.history);
    if (r.custom && r.custom.pick_one) {
      $("r-pick").replaceChildren(...Object.entries(r.custom.pick_one).sort((a, b) => b[1].accuracy - a[1].accuracy).map(([k, v]) => {
        const [d, a] = k.split("/");
        const div = document.createElement("div");
        div.innerHTML = '<div class="head"><b></b><span class="mono"></span></div><div class="track"><i></i><div class="half"></div></div>';
        div.querySelector("b").textContent = (names[d] || d) + " · " + a + " (" + v.options + " options)";
        div.querySelector(".mono").textContent = pct(v.accuracy);
        div.querySelector("i").style.width = (v.accuracy * 100) + "%";
        div.querySelector(".half").style.left = (v.chance * 100) + "%";
        return div;
      }));
    }
    const head = '<tr><th>Domain</th><th>Claim</th><th class="num">Accuracy</th><th class="num">Pairs</th></tr>';'''
    if h.count(old_js) != 1:
        raise SystemExit("Could not find the results loader in frontend/index.html - send me a screenshot.")
    h = h.replace(old_js, new_js)

    curve_fn = '''
function drawCurve(hist) {
  const svg = $("r-curve"), W = 640, H = 260, L = 44, R = 16, T = 16, B = 34;
  const n = hist.length, x = (i) => L + (n === 1 ? 0 : i * (W - L - R) / (n - 1));
  const y = (v) => T + (1 - v) * (H - T - B);
  const path = (key) => hist.map((h, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(h[key]).toFixed(1)).join(" ");
  let out = "";
  for (const t of [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) {
    out += `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" stroke="#e3e6e4"/>`;
    out += `<text x="${L - 6}" y="${y(t) + 4}" text-anchor="end" font-size="11" fill="#6b7471">${Math.round(t * 100)}%</text>`;
  }
  out += `<line x1="${L}" x2="${W - R}" y1="${y(0.5)}" y2="${y(0.5)}" stroke="#2b2f2e" stroke-dasharray="4 4"/>`;
  out += `<path d="${path("train_acc")}" fill="none" stroke="#b5bdb9" stroke-width="2"/>`;
  out += `<path d="${path("val_acc")}" fill="none" stroke="#2563eb" stroke-width="2.5"/>`;
  hist.forEach((h, i) => { out += `<circle cx="${x(i)}" cy="${y(h.val_acc)}" r="3" fill="#2563eb"><title>Epoch ${h.epoch}: validation ${pct(h.val_acc)}, training ${pct(h.train_acc)}</title></circle>`; });
  hist.forEach((h, i) => { if (n <= 16 || i % Math.ceil(n / 12) === 0 || i === n - 1) out += `<text x="${x(i)}" y="${H - 12}" text-anchor="middle" font-size="11" fill="#6b7471">${h.epoch}</text>`; });
  out += `<text x="${(L + W - R) / 2}" y="${H - 1}" text-anchor="middle" font-size="11" fill="#6b7471">epoch</text>`;
  const best = hist.reduce((a, b) => (b.val_acc > a.val_acc ? b : a));
  out += `<text x="${W - R}" y="${y(0.5) + 18}" text-anchor="end" font-size="12" fill="#2563eb">best validation ${pct(best.val_acc)} (epoch ${best.epoch})</text>`;
  svg.innerHTML = out;
}
'''
    h = h.replace("// ---------- results page", curve_fn + "// ---------- results page", 1)
    h = h.replace("Custom dual-encoder model · trained from scratch · no people in the data", "Dual-encoder model · scratch and pretrained variants · no people in the data")
    open(f, "w").write(h)
print("Graphs added to the Model results tab.")
