import torch

torch.manual_seed(0)
RACUNSKI_TIP = torch.int64
INT16_MAKS = 32767

klasteri = []
for n in [5, 3, 7, 4, 6]:
    e = torch.randint(0, n, (2, n * 3), dtype=torch.int64)
    klasteri.append({'num_nodes': n, 'edge_index': e})

potezi64 = [k['edge_index'] for k in klasteri]
potezi16 = [p.to(torch.int16) for p in potezi64]
velicine = [k['num_nodes'] for k in klasteri]

def sastavi(potezi, idx):
    delovi = [potezi[j] for j in idx]
    ei = torch.cat(delovi, dim=1).to(RACUNSKI_TIP)
    brojevi = torch.tensor([p.size(1) for p in delovi])
    of, ofs = 0, []
    for j in idx:
        ofs.append(of)
        of += velicine[j]
    ofseti = torch.tensor(ofs, dtype=RACUNSKI_TIP)
    return ei + torch.repeat_interleave(ofseti, brojevi)

idx = torch.randperm(5).tolist()
a = sastavi(potezi64, idx)
b = sastavi(potezi16, idx)
assert torch.equal(a, b), 'int64 i int16 putanja daju razlicit rezultat'
assert a.dtype == RACUNSKI_TIP
ukupno = sum(velicine[j] for j in idx)
assert a.max() < ukupno and a.min() >= 0
za = int(a.sum().item()); zb = int(b.sum().item())
assert za == zb

granicni = torch.tensor([[INT16_MAKS], [0]], dtype=torch.int16)
assert int(granicni.long().max()) == INT16_MAKS, 'int16 gubi vrednost na granici opsega'

s16 = torch.randint(0, 30000, (2, 700000), dtype=torch.int16).long()
assert int(s16.sum()) > 2**31, 'test zbira ne pokriva prelivanje'
print('PASS: sastavljanje identicno za int64 i int16, opsezi i zbirovi ispravni')
