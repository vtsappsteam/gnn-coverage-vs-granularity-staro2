                      
import csv
import glob
import re
import statistics as st
import sys
from itertools import product
from pathlib import Path

HERE = Path(__file__).parent.parent / 'results'
RA = HERE / "results_a100"
W = ["mali", "standardni", "veliki"]
D = ["L2", "L3", "L4"]
ALFA = 0.05


def seedovi():
    s = []
    for f in sorted(glob.glob(str(RA / "run_*"))):
        m = re.search(r"run_(\w+)$", f)
        if m and (Path(f) / "korak1_graphsage.csv").exists():
            s.append(m.group(1))
    return s


S = seedovi()
K1 = {s: {x["oznaka"]: x for x in csv.DictReader(open(RA / f"run_{s}/korak1_graphsage.csv"))} for s in S}
K2 = {s: {x["oznaka"]: x for x in csv.DictReader(open(RA / f"run_{s}/korak2_clustergcn.csv"))} for s in S
      if (RA / f"run_{s}/korak2_clustergcn.csv").exists()}


def a1(s, o):
    return float(K1[s][o]["naj_test"]) * 100


def a2(s, o):
    return float(K2[s][o]["naj_test"]) * 100


def p_vrednost(razlike):
    n = len(razlike)
    if n == 0:
        return None
    izmereno = abs(st.mean(razlike))
    veci = sum(1 for znaci in product([1, -1], repeat=n)
               if abs(st.mean([z * d for z, d in zip(znaci, razlike)])) >= izmereno - 1e-12)
    return veci / 2 ** n


def najmanja_p(n):
    return 2 / 2 ** n if n else None


def ispitaj(naziv, razlike, opis_a, opis_b):
    p = p_vrednost(razlike)
    sred = st.mean(razlike)
    dev = st.stdev(razlike) if len(razlike) > 1 else 0.0
    isti = all(x > 0 for x in razlike) or all(x < 0 for x in razlike)
    znacajno = p is not None and p <= ALFA
    print(f"\n{naziv}")
    print(f"  {opis_a} naspram {opis_b}")
    print(f"  razlike po izvrsavanju: " + ", ".join(f"{x:+.2f}" for x in razlike))
    print(f"  prosek {sred:+.2f} p.p., std {dev:.2f}, n = {len(razlike)}")
    print(f"  smer isti u svim izvrsavanjima: {'da' if isti else 'ne'}")
    print(f"  p = {p:.4f}  ->  {'ZNACAJNO' if znacajno else 'nije znacajno'} na nivou {ALFA}")
    return {"naziv": naziv, "prosek": sred, "std": dev, "n": len(razlike),
            "p": p, "isti_smer": isti, "znacajno": znacajno}


def samoprovera():
                                                                                              
    assert abs(p_vrednost([1.88, 1.73, 1.96]) - 2 / 8) < 1e-12
                                                                       
    assert abs(p_vrednost([-0.01, 0.65, 0.73]) - 4 / 8) < 1e-12
                                                                                
    assert abs(p_vrednost([1.0, -1.0]) - 1.0) < 1e-12
                                                                                 
    assert abs(p_vrednost([5.0] * 10) - 2 / 2 ** 10) < 1e-12
    assert najmanja_p(3) == 0.25 and najmanja_p(10) == 2 / 1024


def main():
    samoprovera()
    if not S:
        print(f"Nema run_* foldera u {RA}")
        sys.exit(1)
    print(f"Izvrsavanja: {len(S)} (seedovi {', '.join(S)})")
    nm = najmanja_p(len(S))
    print(f"Najmanja moguca dvostrana p vrednost pri {len(S)} izvrsavanja: {nm:.4f}")
    if nm > ALFA:
        print(f"UPOZORENJE: sa {len(S)} izvrsavanja znacajnost na nivou {ALFA} nije dostizna "
              f"ni u principu, jer je {nm:.4f} > {ALFA}. Potrebno je najmanje "
              f"{next(n for n in range(1, 30) if najmanja_p(n) <= ALFA)} izvrsavanja.")

    nalazi = []
    po_dubini = {d: {s: st.mean([a1(s, f"{d}_{w}") for w in W]) for s in S} for d in D}
    nalazi.append(ispitaj("Dubina: dva naspram tri sloja",
                          [po_dubini["L3"][s] - po_dubini["L2"][s] for s in S], "L = 3", "L = 2"))
    nalazi.append(ispitaj("Dubina: tri naspram cetiri sloja",
                          [po_dubini["L4"][s] - po_dubini["L3"][s] for s in S], "L = 4", "L = 3"))

    po_sirini = {w: {s: st.mean([a1(s, f"{d}_{w}") for d in D]) for s in S} for w in W}
    nalazi.append(ispitaj("Sirina: veliki naspram malog uzorka",
                          [po_sirini["veliki"][s] - po_sirini["mali"][s] for s in S],
                          "veliki uzorak", "mali uzorak"))
    nalazi.append(ispitaj("Sirina: standardni naspram malog uzorka",
                          [po_sirini["standardni"][s] - po_sirini["mali"][s] for s in S],
                          "standardni uzorak", "mali uzorak"))

    nalazi.append(ispitaj("Najbolje dve konfiguracije uzorkovanja",
                          [a1(s, "L4_mali") - a1(s, "L3_standardni") for s in S],
                          "L4_mali", "L3_standardni"))

    if K2:
        S2 = [s for s in S if s in K2]
        nalazi.append(ispitaj("Paradigme pri istoj dubini",
                              [a2(s, "K100") - a1(s, "L3_mali") for s in S2],
                              "ClusterGCN K = 100", "GraphSAGE L3_mali"))
        nalazi.append(ispitaj("ClusterGCN pri manjoj memoriji",
                              [a2(s, "K500") - a1(s, "L3_mali") for s in S2],
                              "ClusterGCN K = 500", "GraphSAGE L3_mali"))
        nalazi.append(ispitaj("Najbolji GraphSAGE naspram najboljeg ClusterGCN-a",
                              [max(a1(s, o) for o in K1[s]) - a2(s, "K100") for s in S2],
                              "najbolji GraphSAGE po izvrsavanju", "ClusterGCN K = 100"))
        nalazi.append(ispitaj("ClusterGCN: sitnije naspram krupnijeg particionisanja",
                              [a2(s, "K2000") - a2(s, "K100") for s in S2],
                              "K = 2000", "K = 100"))

    izlaz = RA / "agregat"
    izlaz.mkdir(exist_ok=True)
    with open(izlaz / "test_znacajnosti.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["naziv", "prosek", "std", "n", "p", "isti_smer", "znacajno"])
        w.writeheader()
        w.writerows(nalazi)

    print("\n" + "=" * 70)
    print(f"{'poredjenje':52s} {'prosek':>8s} {'p':>7s}")
    for x in nalazi:
        oznaka = "*" if x["znacajno"] else " "
        print(f"{oznaka} {x['naziv']:50s} {x['prosek']:+7.2f} {x['p']:7.4f}")
    print(f"\nZnacajnih poredjenja: {sum(1 for x in nalazi if x['znacajno'])} od {len(nalazi)}")
    print(f"Snimljeno u {izlaz / 'test_znacajnosti.csv'}")


if __name__ == "__main__":
    main()
