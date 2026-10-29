f = "frontend/index.html"
h = open(f).read()
if "/* compact */" in h:
    raise SystemExit("Compact layout already added.")
css = '''
  /* compact */
  main.wrap { padding:16px 24px 20px; }
  h1 { font-size:22px; }
  .lead { margin:2px 0 10px; font-size:13px; }
  h1 { margin:0; }
  .cols { gap:14px; }
  #page-check .left { display:grid; grid-template-columns: 1fr 1fr; grid-template-rows:auto 1fr; gap:12px; }
  #page-check .left > .card:nth-child(1) { grid-column:2; grid-row:1; }
  #page-check .left > .card:nth-child(2) { grid-column:1; grid-row:1 / span 2; }
  #page-check .left > .card:nth-child(3) { grid-column:2; grid-row:2; }
  @media (max-width: 900px) { #page-check .left { display:flex; flex-direction:column; } }
  .card { padding:14px 16px; }
  .step { margin-bottom:8px; }
  .drop { min-height:260px; }
  .drop img { max-height:300px; }
  .sentence { margin-top:10px; padding:8px 12px; }
  .sentence b { font-size:15px; }
  .go { margin-top:10px; min-height:40px; }
  .chips button { padding:6px 14px; }
  footer { padding:8px 24px; }
  .hint { margin-top:6px; }
</style>'''
if h.count("</style>") != 1:
    raise SystemExit("Could not find the style block - send me a screenshot.")
h = h.replace("</style>", css, 1)
h = h.replace('<span class="mono">01</span><h2>What is in the photo?</h2>', '<span class="mono">02</span><h2>What is in the photo?</h2>')
h = h.replace('<span class="mono">02</span><h2>Upload the photo</h2>', '<span class="mono">01</span><h2>Upload the photo</h2>')
open(f, "w").write(h)
print("Compact layout added. Reload the page with Cmd+Shift+R.")
