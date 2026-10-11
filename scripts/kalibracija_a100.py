import os, time, shutil
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected, degree
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.data

torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage,
])

torch.manual_seed(42)
device = torch.device('cuda')
print('kartica:', torch.cuda.get_device_name(0))
print('jezgara:', os.cpu_count())
shm = shutil.disk_usage('/dev/shm')
print(f'deljena memorija /dev/shm: {shm.total/1e9:.1f} GB', '  (malo, vidi napomenu)' if shm.total < 2e9 else '')

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data')
data = dataset[0]
data.y = data.y.squeeze()
split_idx = dataset.get_idx_split()
data.edge_index = to_undirected(data.edge_index)
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[split_idx['train']] = True
print(f'graf: {data.num_nodes:,} cvorova, {data.num_edges:,} poteza neusmereno')

data = data.to(device)
torch.cuda.synchronize()
OSNOVICA = torch.cuda.memory_allocated() / 1e6
print(f'graf na kartici: {OSNOVICA:.0f} MB')


class GraphSAGE(torch.nn.Module):
    def __init__(self, ulaz, skriveni, izlaz, slojeva, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.slojeva, self.dropout = slojeva, dropout
        self.convs.append(SAGEConv(ulaz, skriveni))
        self.bns.append(torch.nn.BatchNorm1d(skriveni))
        for _ in range(slojeva - 2):
            self.convs.append(SAGEConv(skriveni, skriveni))
            self.bns.append(torch.nn.BatchNorm1d(skriveni))
        self.convs.append(SAGEConv(skriveni, izlaz))

    def forward(self, x, edge_index):
        for i in range(self.slojeva - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def izmeri(num_workers, preciznost='fp32', epoha=5, S=(15, 10, 5), B=512):
    torch.manual_seed(42)
    loader = NeighborLoader(data, num_neighbors=list(S), input_nodes=split_idx['train'],
                            batch_size=B, shuffle=True, num_workers=num_workers)
    model = GraphSAGE(data.num_features, 256, dataset.num_classes, len(S)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    tip = {'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[preciznost]
    scaler = torch.amp.GradScaler('cuda', enabled=(preciznost == 'fp16'))
    vremena, vrh = [], 0.0
    model.train()
    for ep in range(epoha):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            with torch.autocast('cuda', dtype=tip or torch.float32, enabled=(tip is not None)):
                out = model(batch.x, batch.edge_index)[:batch.batch_size]
                maska = batch.train_mask[:batch.batch_size]
                gubitak = F.cross_entropy(out[maska], batch.y[:batch.batch_size][maska])
            scaler.scale(gubitak).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        vremena.append(time.time() - t0)
        vrh = max(vrh, torch.cuda.max_memory_allocated() / 1e6)
    del model, opt, loader
    torch.cuda.empty_cache()
    return float(np.mean(vremena[1:])), vrh - OSNOVICA


print('\n=== BROJ RADNIH PROCESA (L=3, S=[15,10,5], B=512, fp32) ===')
najbolji, najbrze = 0, None
for nw in [0, 4, 8, 16, 32]:
    try:
        t, m = izmeri(nw)
        if najbrze is None or t < najbrze:
            najbrze, najbolji = t, nw
        print(f'  num_workers={nw:3d}: {t:6.2f} s po epohi, {m:6.0f} MB')
    except Exception as e:
        print(f'  num_workers={nw:3d}: greska ({type(e).__name__}: {str(e)[:60]})')

print(f'\n  najbrze: num_workers={najbolji} sa {najbrze:.2f} s po epohi')

print('\n=== PRECIZNOST (sa najboljim brojem radnih procesa) ===')
for p in ['fp32', 'fp16', 'bf16']:
    try:
        t, m = izmeri(najbolji, preciznost=p)
        print(f'  {p}: {t:6.2f} s po epohi, {m:6.0f} MB')
    except Exception as e:
        print(f'  {p}: greska ({type(e).__name__}: {str(e)[:60]})')

print('\n=== PROCENA TRAJANJA CELE DOPUNE ===')
odnos = {'L2_mali': 0.34, 'L2_standardni': 0.42, 'L2_veliki': 0.53, 'L3_mali': 0.55,
         'L3_standardni': 1.00, 'L3_veliki': 1.39, 'L4_mali': 0.96, 'L4_standardni': 1.76,
         'L4_veliki': 2.62}
zbir = sum(odnos.values()) * najbrze
print(f'  Korak 1, 9 konfiguracija po 50 epoha: {zbir*50/60:.0f} min')
print(f'  ceo seed sa Koracima 2 i 4: oko {(zbir*50/60)*1.3:.0f} min')
print(f'  tri seeda: oko {(zbir*50/60)*1.3*3/60:.1f} h')
