                      
import csv
import glob
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent.parent / 'results'
RD = HERE / "results_a100"
OUT = RD / "agregat"

REDOSLED1 = ['L2_mali', 'L2_standardni', 'L2_veliki', 'L3_mali', 'L3_standardni',
             'L3_veliki', 'L4_mali', 'L4_standardni', 'L4_veliki']
REDOSLED2 = ['K100', 'K500', 'K1000', 'K2000']
REDOSLED4 = ['32', '128', '512', '1024', '2048', '4096']


def seedovi():
    s = []
    for f in sorted(glob.glob(str(RD / "run_*"))):
        m = re.search(r"run_(\w+)$", f)
        if m:
            s.append(m.group(1))
    return s


def ucitaj(fajl, kljuc):
    grupe = {}
    for s in seedovi():
        p = RD / f"run_{s}" / fajl
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            grupe.setdefault(r[kljuc], []).append(r)
    return grupe


def ps(vrednosti):
    v = [float(x) for x in vrednosti if x not in ("", None, "None")]
    if not v:
        return None, None
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def fmt(m, d, mult=1.0, dec=2):
    if m is None:
        return "-"
    return f"{m*mult:.{dec}f} ± {(d or 0)*mult:.{dec}f}"


def main():
    s = seedovi()
    if not s:
        print(f"Nema run_* foldera u {RD}")
        sys.exit(1)
    OUT.mkdir(exist_ok=True)
    print(f"Runova: {len(s)} (seedovi {', '.join(s)})\n")

    kartice = set()
    for x in s:
        p = RD / f"run_{x}" / "sazetak.json"
        if p.exists():
            kartice.add(json.load(open(p)).get("kartica"))
    print(f"Kartica: {', '.join(str(k) for k in kartice)}")
    if len(kartice) > 1:
        print("UPOZORENJE: runovi nisu na istoj kartici, vreme i memorija nisu uporedivi")
    print()

             
    g = ucitaj("korak1_graphsage.csv", "oznaka")
    with open(OUT / "korak1_agregat.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["oznaka", "S", "runova", "tacnost_prosek_pct", "tacnost_std_pct",
                    "vreme_prosek_s", "vreme_std_s", "memorija_prosek_mb", "memorija_std_mb"])
        print("KORAK 1: GraphSAGE")
        print(f"  {'konfiguracija':16s} {'tacnost %':>16s} {'s/epohi':>15s} {'memorija MB':>16s}")
        for o in [x for x in REDOSLED1 if x in g]:
            red = g[o]
            a, ad = ps([r["naj_test"] for r in red])
            t, td = ps([r["vreme_po_epohi"] for r in red])
            m, md = ps([r["memorija_mb"] for r in red])
            w.writerow([o, red[0].get("S", ""), len(red), a * 100, ad * 100, t, td, m, md])
            print(f"  {o:16s} {fmt(a, ad, 100):>16s} {fmt(t, td):>15s} {fmt(m, md, dec=0):>16s}")

    print("\n  po dubini:")
    for d in ["L2", "L3", "L4"]:
        v = [float(r["naj_test"]) * 100 for o in g if o.startswith(d) for r in g[o]]
        if v:
            print(f"    {d}: {st.mean(v):.2f} ± {st.stdev(v):.2f} %")
    print("  po sirini:")
    for x in ["mali", "standardni", "veliki"]:
        v = [float(r["naj_test"]) * 100 for o in g if o.endswith(x) for r in g[o]]
        if v:
            print(f"    {x:11s}: {st.mean(v):.2f} ± {st.stdev(v):.2f} %")

             
    g = ucitaj("korak2_clustergcn.csv", "oznaka")
    with open(OUT / "korak2_agregat.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["oznaka", "klastera", "runova", "tacnost_prosek_pct", "tacnost_std_pct",
                    "metis_prosek_s", "metis_std_s", "ms_po_koraku_prosek", "ms_po_koraku_std",
                    "memorija_prosek_mb", "memorija_std_mb"])
        print("\nKORAK 2: ClusterGCN")
        print(f"  {'K':>6s} {'tacnost %':>16s} {'METIS s':>13s} {'ms/korak':>14s} {'memorija MB':>15s}")
        for o in [x for x in REDOSLED2 if x in g]:
            red = g[o]
            a, ad = ps([r["naj_test"] for r in red])
            mt, mtd = ps([r["metis_s"] for r in red])
            t, td = ps([r["vreme_po_koraku_ms"] for r in red])
            m, md = ps([r["memorija_mb"] for r in red])
            w.writerow([o, red[0].get("klastera", ""), len(red), a * 100, ad * 100,
                        mt, mtd, t, td, m, md])
            print(f"  {o:>6s} {fmt(a, ad, 100):>16s} {fmt(mt, mtd, dec=1):>13s} "
                  f"{fmt(t, td):>14s} {fmt(m, md, dec=0):>15s}")

             
    g = ucitaj("korak4_preklapanje.csv", "batch")
    with open(OUT / "korak4_agregat.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "runova", "n_max_naivno", "n_max_stepen",
                    "n_actual_prosek", "n_actual_std", "eta_naivno_prosek_pct", "eta_naivno_std_pct",
                    "eta_stepen_prosek_pct", "eta_stepen_std_pct", "potega_prosek",
                    "memorija_prosek_mb", "memorija_std_mb"])
        print("\nKORAK 4: faktor preklapanja, ogbn-products")
        print(f"  {'B':>6s} {'N_actual':>18s} {'eta naivno %':>16s} {'eta stepen %':>16s} {'memorija MB':>16s}")
        for b in [x for x in REDOSLED4 if x in g]:
            red = g[b]
            n, nd = ps([r["n_actual"] for r in red])
            e1, e1d = ps([r["eta_naivno_pct"] for r in red])
            e2, e2d = ps([r["eta_stepen_pct"] for r in red])
            pt, _ = ps([r["potega"] for r in red])
            m, md = ps([r["memorija_mb"] for r in red])
            w.writerow([b, len(red), red[0]["n_max_naivno"], red[0]["n_max_stepen"],
                        n, nd, e1, e1d, e2, e2d, pt, m, md])
            print(f"  {b:>6s} {fmt(n, nd, dec=0):>18s} {fmt(e1, e1d):>16s} "
                  f"{fmt(e2, e2d):>16s} {fmt(m, md, dec=0):>16s}")

    json.dump({"runova": len(s), "seedovi": s, "kartice": sorted(str(k) for k in kartice)},
              open(OUT / "agregat_meta.json", "w"), indent=2)
    print(f"\nAgregat snimljen u {OUT}/")


if __name__ == "__main__":
    main()
