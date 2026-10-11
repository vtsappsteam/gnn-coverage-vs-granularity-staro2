import torch
print("torch", torch.__version__)
x = torch.randn(10, 4)
src = torch.randn(5, 4)
for dt in [torch.int16, torch.int32, torch.int64]:
    idx1 = torch.tensor([0, 3, 7], dtype=dt)
    idx2 = torch.zeros(5, 4, dtype=dt)
    rows = []
    for name, fn in [
        ("advanced indexing x[idx]", lambda: x[idx1]),
        ("index_select", lambda: torch.index_select(x, 0, idx1)),
        ("gather", lambda: torch.gather(x, 0, idx2)),
        ("scatter_add_", lambda: torch.zeros(10, 4).scatter_add_(0, idx2, src)),
        ("scatter_reduce", lambda: torch.zeros(10, 4).scatter_reduce(0, idx2, src, reduce="sum")),
        ("embedding", lambda: torch.nn.functional.embedding(idx1, x)),
        ("index_add_", lambda: torch.zeros(10, 4).index_add_(0, idx1, x[:3])),
    ]:
        try:
            fn()
            rows.append((name, "RADI"))
        except Exception as e:
            rows.append((name, "ODBIJA: " + str(e).split(".")[0][:80]))
    print()
    print(str(dt))
    for name, r in rows:
        print(f"  {name:28s} {r}")
