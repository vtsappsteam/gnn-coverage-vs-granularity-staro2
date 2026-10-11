import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.parent / 'results'
REZ = HERE / 'rezultati_int16'
SEEDOVI = [42, 123, 7, 1, 2, 3, 4, 5, 6, 8]
KLASTERI = [100, 500, 2000]
VARIJANTE = ['osnova', 'izolovano_int64', 'izolovano_int16']


def ucitaj():
    podaci = {}
    for s in SEEDOVI:
        with open(REZ / f'run_{s}' / 'int16.csv') as f:
            podaci[s] = {r['oznaka']: r for r in csv.DictReader(f)}
    ocekivano = {f'K{k}_{v}' for k in KLASTERI for v in VARIJANTE}
    for s in SEEDOVI:
        fali = ocekivano - set(podaci[s])
        assert not fali, f'seed {s} nema {fali}'
    return podaci


def main():
    podaci = ucitaj()

    problemi = 0
    for s in SEEDOVI:
        for k in KLASTERI:
            a = podaci[s][f'K{k}_izolovano_int64']
            b = podaci[s][f'K{k}_izolovano_int16']
            for kol in ['kontrola_cvorova', 'kontrola_potega', 'kontrola_zbir']:
                if a[kol] != b[kol]:
                    problemi += 1
    assert problemi == 0, f'sekvence batch-eva se razlikuju u {problemi} kontrola'
    print('identitet sekvence batch-eva: PROSLO u svih 30 poredjenja')

    izlaz = REZ / 'agregat'
    izlaz.mkdir(exist_ok=True)
    redovi = []
    for k in KLASTERI:
        for v in VARIJANTE:
            t = [float(podaci[s][f'K{k}_{v}']['naj_test']) * 100 for s in SEEDOVI]
            m = [float(podaci[s][f'K{k}_{v}']['memorija_mb']) for s in SEEDOVI]
            vr = [float(podaci[s][f'K{k}_{v}']['vreme_po_koraku_ms']) for s in SEEDOVI]
            nk = [int(podaci[s][f'K{k}_{v}']['naj_korak']) for s in SEEDOVI]
            strukture = {podaci[s][f'K{k}_{v}']['struktura_mb'] for s in SEEDOVI}
            tipovi = {podaci[s][f'K{k}_{v}']['tip_racunanja'] for s in SEEDOVI}
            assert len(strukture) == 1 and len(tipovi) == 1
            redovi.append({
                'klastera': k, 'varijanta': v, 'izvrsavanja': len(SEEDOVI),
                'tip_racunanja': tipovi.pop(),
                'tacnost_prosek': np.mean(t), 'tacnost_std': np.std(t, ddof=1),
                'memorija_prosek_mb': np.mean(m), 'memorija_std_mb': np.std(m, ddof=1),
                'vreme_prosek_ms': np.mean(vr), 'vreme_std_ms': np.std(vr, ddof=1),
                'struktura_mb': float(strukture.pop()),
                'naj_korak_prosek': np.mean(nk),
                'vrh_u_poslednjoj_oceni': sum(1 for x in nk if x == 5000)})

    with open(izlaz / 'trening_agregat.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(redovi[0].keys()))
        w.writeheader()
        w.writerows(redovi)

    razlike = [abs(float(podaci[s][f'K{k}_izolovano_int16']['naj_test'])
                   - float(podaci[s][f'K{k}_izolovano_int64']['naj_test'])) * 100
               for s in SEEDOVI for k in KLASTERI]
    print(f'najveca razlika tacnosti int16 naspram int64: {max(razlike):.2f} p.p.')

    for r in redovi:
        print(f"K={r['klastera']:<5d}{r['varijanta']:16s} "
              f"tacnost {r['tacnost_prosek']:.2f} +/- {r['tacnost_std']:.2f} %  "
              f"vrh {r['memorija_prosek_mb']:.0f} MB  "
              f"{r['vreme_prosek_ms']:.1f} ms/korak  "
              f"struktura {r['struktura_mb']:.1f} MB")
    print(f"upisano: {izlaz / 'trening_agregat.csv'}")


if __name__ == '__main__':
    main()
