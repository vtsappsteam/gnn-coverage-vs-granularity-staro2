import csv
import statistics as st
from itertools import product
from pathlib import Path

HERE = Path(__file__).parent.parent / 'results'
S = [42, 123, 7, 1, 2, 3, 4, 5, 6, 8]


def p_upareno(d):
    obs = abs(st.mean(d))
    svi = [abs(st.mean([x * z for x, z in zip(d, zn)])) for zn in product([1, -1], repeat=len(d))]
    return sum(1 for v in svi if v >= obs - 1e-12) / len(svi)


def main():
    R = {s: list(csv.DictReader(open(HERE / f'rezultati_rns/run_{s}/rns.csv')))[0] for s in S}
    I = {s: {r['oznaka']: r for r in csv.DictReader(open(HERE / f'rezultati_int16/run_{s}/int16.csv'))} for s in S}
    izlaz = HERE / 'rezultati_rns' / 'agregat'
    izlaz.mkdir(exist_ok=True)
    t = [float(R[s]['naj_test']) * 100 for s in S]
    red = {'oznaka': R[42]['oznaka'], 'izvrsavanja': len(S), 'cvorova_po_koraku': float(R[42]['cvorova_po_koraku']),
           'potega_po_koraku': st.mean(float(R[s]['potega_po_koraku']) for s in S),
           'test_prosek': st.mean(t), 'test_std': st.stdev(t),
           'memorija_prosek_mb': st.mean(float(R[s]['memorija_mb']) for s in S),
           'vreme_prosek_ms': st.mean(float(R[s]['vreme_po_koraku_ms']) for s in S),
           'energija_prosek_j': st.mean(float(R[s]['energija_j']) for s in S)}
    testovi = []
    for naziv, kl in [('RNS naspram ClusterGCN osnova K100', 'K100_osnova'), ('RNS naspram izolovano int64 K100', 'K100_izolovano_int64')]:
        d = [(float(R[s]['naj_test']) - float(I[s][kl]['naj_test'])) * 100 for s in S]
        testovi.append({'poredjenje': naziv, 'prosek_pp': st.mean(d), 'std_pp': st.stdev(d), 'p': p_upareno(d),
                        'pozitivnih': sum(1 for x in d if x > 0), 'negativnih': sum(1 for x in d if x < 0)})
    with open(izlaz / 'rns_agregat.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(red.keys()))
        w.writeheader()
        w.writerow(red)
    with open(izlaz / 'znacajnost_rns.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(testovi[0].keys()))
        w.writeheader()
        w.writerows(testovi)
    print(f"RNS test {red['test_prosek']:.2f} +/- {red['test_std']:.2f}, potega {red['potega_po_koraku']:.0f}, vrh {red['memorija_prosek_mb']:.0f} MB")
    for x in testovi:
        print(f"{x['poredjenje']:40s} {x['prosek_pp']:+.2f} pp p={x['p']:.4f} (+{x['pozitivnih']}/-{x['negativnih']})")


if __name__ == '__main__':
    main()
