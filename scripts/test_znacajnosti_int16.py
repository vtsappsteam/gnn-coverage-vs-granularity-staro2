import csv
from itertools import product
from pathlib import Path

import numpy as np

from agregacija_int16 import ucitaj, SEEDOVI, KLASTERI, REZ


def p_upareno(d):
    obs = abs(np.mean(d))
    svi = [abs(np.mean([x * z for x, z in zip(d, znak)])) for znak in product([1, -1], repeat=len(d))]
    return sum(1 for v in svi if v >= obs - 1e-12) / len(svi)


def main():
    podaci = ucitaj()
    redovi = []

    def dodaj(poredjenje, velicina, d):
        redovi.append({'poredjenje': poredjenje, 'velicina': velicina,
                       'prosek': np.mean(d), 'std': np.std(d, ddof=1),
                       'p': p_upareno(d), 'pozitivnih': sum(1 for x in d if x > 0),
                       'negativnih': sum(1 for x in d if x < 0)})

    for k in KLASTERI:
        d = [(float(podaci[s][f'K{k}_osnova']['naj_test'])
              - float(podaci[s][f'K{k}_izolovano_int64']['naj_test'])) * 100 for s in SEEDOVI]
        dodaj(f'K{k} osnova naspram izolovano_int64', 'tacnost_pp', d)
    for k in KLASTERI:
        d = [(float(podaci[s][f'K{k}_izolovano_int16']['naj_test'])
              - float(podaci[s][f'K{k}_izolovano_int64']['naj_test'])) * 100 for s in SEEDOVI]
        dodaj(f'K{k} izolovano_int16 naspram izolovano_int64', 'tacnost_pp', d)
    for k in KLASTERI:
        d = [float(podaci[s][f'K{k}_izolovano_int64']['vreme_po_koraku_ms'])
             - float(podaci[s][f'K{k}_izolovano_int16']['vreme_po_koraku_ms']) for s in SEEDOVI]
        dodaj(f'K{k} vreme int64 naspram int16', 'ms_po_koraku', d)

    izlaz = REZ / 'agregat' / 'znacajnost_int16.csv'
    izlaz.parent.mkdir(exist_ok=True)
    with open(izlaz, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(redovi[0].keys()))
        w.writeheader()
        w.writerows(redovi)

    for r in redovi:
        print(f"{r['poredjenje']:48s} {r['prosek']:+.2f} ({r['velicina']}), p={r['p']:.4f}")
    print(f'upisano: {izlaz}')


if __name__ == '__main__':
    main()
