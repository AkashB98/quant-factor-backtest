"""Zero-dependency web UI: equity curves, drawdowns, rebalance log.

Run:  python server.py [--port 8000]
Then open http://localhost:8000 in a browser.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from backtest import Config, run, run_equal_weight
from data_gen import load_universe, write_universe

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FACTORS = ["momentum", "reversal", "lowvol"]

try:
    PANEL = load_universe(DATA)
except FileNotFoundError:
    write_universe(DATA)
    PANEL = load_universe(DATA)

RESULTS = {f: run(PANEL, Config(factor=f, cost_bps=10.0)) for f in FACTORS}
RESULTS["equal_weight"] = run_equal_weight(PANEL)

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Quant Factor Backtest</title>
<style>
body{font-family:system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#1a1a1a}
h1{font-size:1.4em} .warn{background:#fff8e1;border:1px solid #f0c36d;padding:8px 12px;border-radius:6px;font-size:.85em}
canvas{width:100%;height:260px;border:1px solid #ddd;border-radius:6px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:.85em;margin:12px 0}
th,td{border:1px solid #ddd;padding:6px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}
.tabs button{margin-right:6px;padding:6px 12px;border:1px solid #ccc;background:#f7f7f7;border-radius:4px;cursor:pointer}
.tabs button.active{background:#1a1a1a;color:#fff}
</style></head><body>
<h1>Quant Factor Backtester</h1>
<div class="warn"><b>SIMULATED DATA</b> — every price is synthetically generated. Not real market data, not investment advice.</div>
<div class="tabs" id="tabs"></div>
<canvas id="equity" width="920" height="260"></canvas>
<canvas id="dd" width="920" height="260"></canvas>
<h2>Headline metrics</h2>
<table id="metrics"><thead><tr><th>metric</th><th>value</th></tr></thead><tbody></tbody></table>
<h2>Rebalance log (last 12)</h2>
<table id="rebal"><thead><tr><th>date</th><th>n_long</th><th>n_short</th><th>turnover</th><th>cost</th><th>equity</th></tr></thead><tbody></tbody></table>
<script>
const COLORS={momentum:"#2563eb",reversal:"#dc2626",lowvol:"#059669",equal_weight:"#6b7280"};
let current="momentum";
async function load(name){
  current=name;
  document.querySelectorAll(".tabs button").forEach(b=>b.classList.toggle("active",b.dataset.n===name));
  const r=await fetch("/api/result/"+name).then(r=>r.json());
  draw(document.getElementById("equity"),r.dates,r.equity,"Equity curve (net of costs)",true);
  const dd=r.equity.map((v,i)=>{const peak=Math.max(...r.equity.slice(0,i+1));return (v-peak)/peak*100;});
  draw(document.getElementById("dd"),r.dates,dd,"Drawdown %",false);
  const s=r.summary;
  const rows=[["total return",(s.total_return*100).toFixed(1)+"%"],["CAGR",(s.cagr*100).toFixed(1)+"%"],
    ["Sharpe",s.sharpe.toFixed(2)],["Sortino",s.sortino.toFixed(2)],
    ["max drawdown",(s.max_drawdown*100).toFixed(1)+"%"],["win rate",(s.win_rate*100).toFixed(1)+"%"],
    ["avg turnover/rebalance",r.avg_turnover.toFixed(2)],["total cost drag",(r.total_cost_drag*100).toFixed(2)+"%"]];
  document.querySelector("#metrics tbody").innerHTML=rows.map(x=>`<tr><td>${x[0]}</td><td>${x[1]}</td></tr>`).join("");
  document.querySelector("#rebal tbody").innerHTML=r.rebalances.slice(-12).reverse()
    .map(x=>`<tr><td>${x.date}</td><td>${x.n_long}</td><td>${x.n_short}</td><td>${x.turnover}</td><td>${x.cost}</td><td>${x.equity}</td></tr>`).join("");
}
function draw(cv,dates,vals,title,log){
  const ctx=cv.getContext("2d"),W=cv.width,H=cv.height;ctx.clearRect(0,0,W,H);
  const mn=Math.min(...vals),mx=Math.max(...vals),pad=(mx-mn)||1;
  ctx.fillStyle="#111";ctx.font="13px system-ui";ctx.fillText(title,10,18);
  ctx.strokeStyle=COLORS[current];ctx.lineWidth=1.5;ctx.beginPath();
  vals.forEach((v,i)=>{const x=40+i*(W-60)/(vals.length-1),y=H-20-(v-mn)/pad*(H-50);
    i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  ctx.stroke();
  ctx.fillStyle="#666";ctx.fillText(mx.toFixed(2),W-52,24);ctx.fillText(mn.toFixed(2),W-52,H-8);
}
const tabs=document.getElementById("tabs");
["momentum","reversal","lowvol","equal_weight"].forEach(n=>{
  const b=document.createElement("button");b.textContent=n;b.dataset.n=n;
  b.onclick=()=>load(n);tabs.appendChild(b);});
load("momentum");
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path.startswith("/api/result/"):
            name = path.rsplit("/", 1)[-1]
            if name not in RESULTS:
                self.send_error(404)
                return
            r = RESULTS[name]
            self._json({
                "factor": r.factor,
                "dates": [d.isoformat() for d in r.dates],
                "equity": [round(v, 6) for v in r.equity],
                "summary": {k: round(v, 6) if isinstance(v, float) else v
                            for k, v in r.summary.items()},
                "avg_turnover": round(r.avg_turnover, 4),
                "total_cost_drag": round(r.total_cost_drag, 6),
                "rebalances": r.rebalances,
            })
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
