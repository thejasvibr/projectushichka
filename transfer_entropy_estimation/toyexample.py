from pyinform import transfer_entropy
xs = [0,1,1,1,1,0,0,0,0]
ys = [0,0,1,1,1,1,0,0,0]
print(transfer_entropy(xs, ys, k=2))

print(transfer_entropy(xs, ys, k=3))
