import csv
import statistics as st
from itertools import product
from pathlib import Path

HERE = Path(__file__).parent.parent / 'results'
REZ = HERE / 'rezultati_pokrivenost_gustina'
S = [42, 123, 7, 1, 2, 3, 4, 5, 6, 8]
GRAFOVI = ['Reddit', 'products']


def p_upareno(d):
    obs = abs(st.mean(d))
    svi = [abs(st.mean([x * z for x, z in zip(d, zn)])) for zn in product([1, -1], repeat=len(d))]
    return sum(1 for v in svi if v >= obs - 1e-12) / len(svi)


def ucitaj(graf):
    return {s: {r['oznaka']: r for r in csv.DictReader(open(REZ / graf / f'run_{s}' / 'pokrivenost.csv'))} for s in S}


def main():
    izlaz = REZ / 'agregat'
    izlaz.mkdir(exist_ok=True)
    redovi, testovi = [], []
    for graf in GRAFOVI:
        D = ucitaj(graf)
        ocekivano = set(D[42])
        for s in S:
            assert set(D[s]) == ocekivano, f'{graf} seed {s} nepotpun'
        for o in D[42]:
            t = [float(D[s][o]['test_tacno']) * 100 for s in S]
            v = [float(D[s][o]['val_tacno']) * 100 for s in S]
            m = [float(D[s][o]['memorija_mb']) for s in S]
            ms = [float(D[s][o]['vreme_po_koraku_ms']) for s in S]
            e = [float(D[s][o]['energija_j']) for s in S]
            w = [float(D[s][o]['snaga_prosek_w']) for s in S]
            redovi.append({
                'graf': graf, 'oznaka': o, 'grana': D[42][o]['grana'], 'klastera': int(D[42][o]['klastera']),
                'batch_klastera': int(D[42][o]['batch_klastera']), 'pokrivenost_pct': float(D[42][o]['pokrivenost_pct']),
                'izvrsavanja': len(S), 'test_prosek': st.mean(t), 'test_std': st.stdev(t),
                'val_prosek': st.mean(v), 'val_std': st.stdev(v),
                'cvorova_po_koraku': st.mean(float(D[s][o]['cvorova_po_koraku']) for s in S),
                'potega_po_koraku': st.mean(float(D[s][o]['potega_po_koraku']) for s in S),
                'memorija_prosek_mb': st.mean(m), 'memorija_std_mb': st.stdev(m),
                'vreme_prosek_ms': st.mean(ms), 'vreme_std_ms': st.stdev(ms),
                'energija_prosek_j': st.mean(e), 'energija_std_j': st.stdev(e), 'snaga_prosek_w': st.mean(w),
                'vrh_u_poslednjoj_oceni': sum(1 for s in S if int(D[s][o]['naj_korak']) == 2500)})
        for ime, a, b in [('fiksno K200 naspram K2000', 'fiksno_K200', 'fiksno_K2000'),
                          ('konstantna pokrivenost K200 naspram K2000', 'fiksno_K200', 'pokrivenost_K2000'),
                          ('konstantna pokrivenost K200 naspram K500', 'fiksno_K200', 'pokrivenost_K500'),
                          ('konstantna pokrivenost K200 naspram K1000', 'fiksno_K200', 'pokrivenost_K1000')]:
            d = [(float(D[s][a]['test_tacno']) - float(D[s][b]['test_tacno'])) * 100 for s in S]
            testovi.append({'graf': graf, 'poredjenje': ime, 'prosek_pp': st.mean(d), 'std_pp': st.stdev(d),
                            'p': p_upareno(d), 'pozitivnih': sum(1 for x in d if x > 0), 'negativnih': sum(1 for x in d if x < 0)})
    with open(izlaz / 'pokrivenost_agregat.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(redovi[0].keys()))
        w.writeheader()
        w.writerows(redovi)
    with open(izlaz / 'znacajnost_pokrivenost.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(testovi[0].keys()))
        w.writeheader()
        w.writerows(testovi)
    for r in redovi:
        print(f"{r['graf']:8s} {r['oznaka']:20s} test {r['test_prosek']:6.2f} +/- {r['test_std']:4.2f}  vrh {r['memorija_prosek_mb']:6.0f} MB  {r['energija_prosek_j']:6.0f} J")
    for t in testovi:
        print(f"{t['graf']:8s} {t['poredjenje']:44s} {t['prosek_pp']:+.2f} pp  p={t['p']:.4f}  (+{t['pozitivnih']}/-{t['negativnih']})")
    print(f"upisano: {izlaz}")


if __name__ == '__main__':
    main()
