                      
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent.parent / 'results'
RD = HERE / "results_prosirenje_a100"
OUT = RD / "agregat"
GRAFOVI = ["arxiv", "yelp", "products", "reddit"]
FANOUTI = ["mali", "standardni", "veliki"]
BATCHEVI = [32, 128, 512, 1024, 2048, 4096]
PRECIZNOSTI = ["fp32", "fp16", "bf16"]


def broj(x):
    return float(x) if x not in ("", None, "None") else None


def main():
    OUT.mkdir(exist_ok=True)
    f1 = list(csv.DictReader(open(RD / "faza1.csv")))
    fs = list(csv.DictReader(open(RD / "faza1_stepen.csv")))
    f2 = list(csv.DictReader(open(RD / "faza2.csv")))
    print(f"faza1 {len(f1)} redova, faza1_stepen {len(fs)}, faza2 {len(f2)}\n")

                     
    vid = {}
    for x in fs:
        vid.setdefault(x["graf"], x)
    with open(OUT / "grafovi.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graf", "cvorova", "potega", "atributa", "prosecan_stepen",
                    "medijana_stepena", "najveci_stepen"])
        print("OSOBINE GRAFOVA")
        print(f"  {'graf':10s} {'cvorova':>11s} {'potega':>13s} {'atributa':>9s} {'stepen':>8s}")
        for g in GRAFOVI:
            x = vid[g]
            w.writerow([g, x["cvorova"], x["potega"], x["atributa"],
                        x["prosecan_ulazni_stepen"], x["medijana_stepena"], x["najveci_stepen"]])
            print(f"  {g:10s} {int(x['cvorova']):11,d} {int(x['potega']):13,d} "
                  f"{int(x['atributa']):9d} {float(x['prosecan_ulazni_stepen']):8.2f}")

                               
    with open(OUT / "granice.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graf", "fanout", "granica_naivna", "granica_sa_stepenom", "odnos",
                    "udeo_ispod_S1_pct", "efektivni_fanout"])
        print("\nGRANICA PO CILJANOM CVORU")
        print(f"  {'graf':10s} {'fanout':12s} {'naivna':>8s} {'stepen':>8s} {'odnos':>7s} {'% ispod':>9s}")
        for x in fs:
            n, s = float(x["granica_naivna"]), float(x["granica_sa_stepenom"])
            w.writerow([x["graf"], x["fanout"], n, s, n / s,
                        x["udeo_ispod_S1_pct"], x["efektivni_fanout"]])
            print(f"  {x['graf']:10s} {x['fanout']:12s} {n:8.0f} {s:8.0f} {n/s:6.2f}x "
                  f"{float(x['udeo_ispod_S1_pct']):8.1f}%")

                                                      
    kljuc = {(x["graf"], x["fanout"], int(x["batch"]), x["preciznost"]): x for x in f1}
    with open(OUT / "preklapanje.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graf", "fanout", "batch", "n_actual", "eta_naivno_pct", "eta_stepen_pct",
                    "udeo_pravog_pct", "potega", "memorija_fp32_mb", "memorija_fp16_mb",
                    "memorija_bf16_mb", "usteda_fp16", "ms_po_batchu_fp32"])
        for g in GRAFOVI:
            for fo in FANOUTI:
                for b in BATCHEVI:
                    x = kljuc.get((g, fo, b, "fp32"))
                    if not x or x["oom"] == "True":
                        continue
                    m32 = broj(x["memorija_mb"])
                    m16 = broj(kljuc[(g, fo, b, "fp16")]["memorija_mb"])
                    mbf = broj(kljuc[(g, fo, b, "bf16")]["memorija_mb"])
                    en, es = broj(x["eta_naivno_pct"]), broj(x["eta_stepen_pct"])
                    w.writerow([g, fo, b, broj(x["n_actual"]), en, es,
                                es / en * 100 if en else None, broj(x["potega"]),
                                m32, m16, mbf, m32 / m16 if m16 else None,
                                broj(x["ms_po_batchu"])])

    print("\nUSTEDA MEMORIJE od 16-bitne preciznosti (prosek po grafu)")
    print(f"  {'graf':10s} {'fp32/fp16':>11s} {'fp32/bf16':>11s}")
    with open(OUT / "preciznost.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graf", "odnos_fp16", "odnos_bf16", "konfiguracija"])
        for g in GRAFOVI:
            o16, obf = [], []
            for fo in FANOUTI:
                for b in BATCHEVI:
                    x = kljuc.get((g, fo, b, "fp32"))
                    if not x or x["oom"] == "True":
                        continue
                    m32 = broj(x["memorija_mb"])
                    m16 = broj(kljuc[(g, fo, b, "fp16")]["memorija_mb"])
                    mbf = broj(kljuc[(g, fo, b, "bf16")]["memorija_mb"])
                    o16.append(m32 / m16)
                    obf.append(m32 / mbf)
                    w.writerow([g, m32 / m16, m32 / mbf, f"{fo}/B{b}"])
            print(f"  {g:10s} {st.mean(o16):11.2f} {st.mean(obf):11.2f}")

            
    gr = defaultdict(list)
    for x in f2:
        gr[(x["graf"], x["sta"], x["vrednost"])].append((float(x["ms_po_batchu"]),
                                                        float(x["memorija_mb"])))
    with open(OUT / "optimizacije.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graf", "sta", "vrednost", "ponavljanja", "ms_prosek", "ms_std",
                    "memorija_prosek_mb", "memorija_std_mb"])
        print("\nSISTEMSKE OPTIMIZACIJE")
        for g in ["arxiv", "products"]:
            print(f"  {g}:")
            for sta, vrednosti in [("radnih_procesa", ["0", "2", "4", "8", "16", "32"]),
                                   ("tf32", ["False", "True"]),
                                   ("compile", ["False", "True"])]:
                for v in vrednosti:
                    red = gr[(g, sta, v)]
                    if not red:
                        continue
                    t = [x[0] for x in red]
                    m = [x[1] for x in red]
                    ts = st.stdev(t) if len(t) > 1 else 0.0
                    ms = st.stdev(m) if len(m) > 1 else 0.0
                    w.writerow([g, sta, v, len(red), st.mean(t), ts, st.mean(m), ms])
                    print(f"    {sta:16s} {v:>6s}: {st.mean(t):8.1f} ± {ts:5.1f} ms, "
                          f"{st.mean(m):8.0f} ± {ms:4.0f} MB")

    sz = json.load(open(RD / "sazetak.json"))
    json.dump({"kartica": sz.get("kartica"), "jezgara": sz.get("jezgara"),
               "deljena_memorija_gb": sz.get("deljena_memorija_gb"),
               "grafova": len(GRAFOVI), "merenja_faza1": len(f1), "merenja_faza2": len(f2)},
              open(OUT / "agregat_meta.json", "w"), indent=2)
    print(f"\nAgregat snimljen u {OUT}/")


if __name__ == "__main__":
    main()
