import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.parent / 'results'
IZVOR = HERE / 'results_prosirenje_a100' / 'faza1.csv'
IZLAZ = HERE / 'rezultati_lanac'
IZLAZ.mkdir(exist_ok=True)

redovi = [r for r in csv.DictReader(open(IZVOR))
          if r['preciznost'] == 'fp32' and r['oom'] == 'False']

rezultat = []
for graf in sorted({r['graf'] for r in redovi}):
    za_graf = [r for r in redovi if r['graf'] == graf]
    n_pred = np.array([float(r['n_max_stepen']) for r in za_graf])
    n_meri = np.array([float(r['n_actual']) for r in za_graf])
    mem = np.array([float(r['memorija_mb']) for r in za_graf])
    batch = np.array([int(r['batch']) for r in za_graf])

    nagib, odsecak = np.polyfit(n_meri, mem, 1)
    fit = nagib * n_meri + odsecak
    r2 = 1 - ((mem - fit) ** 2).sum() / ((mem - mem.mean()) ** 2).sum()

    mem_pred = nagib * n_pred + odsecak
    for i, r in enumerate(za_graf):
        rezultat.append({
            'graf': graf, 'fanout': r['fanout'], 'batch': int(batch[i]),
            'n_predvidjeno': n_pred[i], 'n_izmereno': n_meri[i],
            'greska_cvorova_pct': (n_pred[i] - n_meri[i]) / n_meri[i] * 100,
            'mem_predvidjeno_mb': mem_pred[i], 'mem_izmereno_mb': mem[i],
            'greska_mem_pct': (mem_pred[i] - mem[i]) / mem[i] * 100,
            'nagib_kb_po_cvoru': nagib * 1000, 'odsecak_mb': odsecak, 'r2_linearnosti': r2})

with open(IZLAZ / 'lanac_predikcije.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rezultat[0].keys()))
    w.writeheader()
    w.writerows(rezultat)

print(f"{'graf':10s} {'nagib KB/cvor':>14s} {'R2':>7s} {'greska cvorova min..max %':>28s} {'greska memorije min..max %':>28s}")
for graf in sorted({r['graf'] for r in rezultat}):
    g = [r for r in rezultat if r['graf'] == graf]
    gc = [r['greska_cvorova_pct'] for r in g]
    gm = [r['greska_mem_pct'] for r in g]
    print(f"{graf:10s} {g[0]['nagib_kb_po_cvoru']:14.1f} {g[0]['r2_linearnosti']:7.4f} "
          f"{min(gc):13.1f} .. {max(gc):8.1f} {min(gm):14.1f} .. {max(gm):8.1f}")

print()
print('greska cvorova po velicini batch-a (prosek preko grafova i fanouta):')
for b in sorted({r['batch'] for r in rezultat}):
    g = [r['greska_cvorova_pct'] for r in rezultat if r['batch'] == b]
    print(f'  B={b:5d}: {np.mean(g):+6.1f} %')
print(f"upisano: {IZLAZ / 'lanac_predikcije.csv'}")
