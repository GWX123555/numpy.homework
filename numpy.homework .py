import numpy as np

#1
print("NP-E1")
A=np.array([
    [0.20,-0.50],[1.10,0.40],
    [0.80, 0.30],[-0.70,0.60],
    [-0.40,0.90],[0.10,-0.20]
])
print("A.ndim=",A.ndim)
print("A.shape =", A.shape)
print("A.size =", A.size)

row_1 = A[1, :]
print("\n第2行（下标1）：", row_1)
print("第2行 shape：", row_1.shape)
last_two_cols = A[:, -2:]
print("\n最后2列：\n", last_two_cols)
print("最后2列 shape：", last_two_cols.shape)
B = A.copy()
B[B < 0] = 0
print("\nB（负数替换为0）：\n", B)
print("原A不变：\n", A)

#2

print("\nNP-E2")
a = np.array([0.80, -0.60, 0.70])
b = np.array([-0.20, 0.90, 0.40])

 
mul_elem = a * b
dot_ab = a @ b
print("a * b（逐元素相乘）：", mul_elem)
print("a*b shape：", mul_elem.shape)
print("a @ b（点积）：", dot_ab)
print("a@b shape：", np.shape(dot_ab))

 
len_a = np.sqrt(np.sum(a ** 2))
len_b = np.sqrt(np.sum(b ** 2))
print("\n手动计算 a 的长度：", len_a)
print("手动计算 b 的长度：", len_b)

 
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
print("\nnp.linalg.norm 核对 a：", norm_a)
print("np.linalg.norm 核对 b：", norm_b)


#3
print("\nNP-E3")
def angle_deg(a, b):
    
    if a.ndim !=1 or b.ndim !=1:
        raise ValueError("必须一维向量")
    if a.shape != b.shape:
        raise ValueError("维度要相同")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb ==0:
        raise ValueError("不能是零向量")
    cos_theta = (a @ b) / (na * nb)
    cos_theta = np.clip(cos_theta, -1, 1)
    rad = np.arccos(cos_theta)
    return np.degrees(rad)

a = np.array([1.00, -1.00])
b = np.array([1.00, 0.00])
print(angle_deg(a,b))



#4
print("\nNP-E4")
p = np.array([-0.50, 1.20, 2.00])
q = np.array([0.80, -0.40, 1.10])

diff = p - q
euclid = np.linalg.norm(diff)
manhattan = np.sum(np.abs(diff))

print("差向量：", diff)
print("欧氏距离：", euclid)
print("曼哈顿距离：", manhattan)


#5
print("\nNP-E5")
def unit_vector(a):
    if a.ndim != 1:
        raise ValueError("只能一维向量")
    n = np.linalg.norm(a)
    if n == 0:
        raise ValueError("零向量不能归一化")
    return a / n   


a = np.array([-1.20, 0.40, 0.80])
u = unit_vector(a)
print("单位向量：", u)
print(np.allclose(np.linalg.norm(u), 1)) 


#6
print("\nNP-E6")
A = np.array([
    [0.20, -0.50, 1.10],
    [0.80,  0.40, -0.30]
])
print(A.shape)  
s0 = A.sum(axis=0)
print(s0)
print(s0.shape) # (3,)
m1 = A.mean(axis=1)
print(m1)
print(m1.shape) # (2,)

max_all = A.max()
print(max_all)

#7
print("\nNP-E7")

X = np.array([
    [0.80, -0.60, 0.70],
    [-0.40, 0.50, 0.20]
])
scale = np.array([1.50, 0.50, -1.00])
bias = np.array([-0.10, 0.20, 0.30])

Y = X * scale + bias
print(Y)
print(Y.shape) # (2,3)


#8
print("\nNP-E8")
a = np.array([
    [[0.20, -0.40], [0.70, 0.10]],
    [[0.30, 0.80], [-0.20, 0.50]],
    [[-0.10, 0.60], [0.40, -0.30]]
])
b = a + 0.10
print(a.shape) # (3,2,2)   C=3通道，H=2，W=2
batch = np.stack([a, b])
print(batch.shape) # (2, 3, 2, 2)
blue_chan = batch[:, 2, :, :]
print(blue_chan.shape) # (2, 2, 2)
batch_hwc = batch.transpose(0, 2, 3, 1)
print(batch_hwc.shape) # (2, 2, 2, 3)

#9
def angles_to(query,X):
    q_norm=np.linalg.norm(query)
    if np.isclose(q_norm,0):
        raise ValueError("query不能是零向量")
    x_norms=np.linalg.norm(X,axis=1)
    dot=X @ query
    cos_theta=dot/(x_norms*q_norm)
    cos_theta=np.clip(cos_theta,-1.0,1.0)
    angles=np.degrees(np.arccos(cos_theta))
    angles[np.isclose(x_norms,0)]=np.nan
    return angles
if __name__=="__main__":
    query = np.array([0.60, -0.80, 0.00])
    X = np.array([
        [0.30, -0.40, 0.00],
        [-0.60, 0.80, 0.00],
        [0.80, 0.60, 0.00],
        [0.00, 0.00, 0.00]
    ])
    ang=angles_to(query,X)
    print("M1夹角：", ang)

#10
def top_k_cosine(query,gallery,k):
    q_norm=np.linalg.norm(query)
    if np.isclose(q_norm,0):
        raise ValueError("query不能是零向量")
    g_norms=np.linalg.norm(gallery,axis=1)
    dot=gallery @ query
    sim=dot/(g_norms*q_norm)
    vaild_mask=~np.isclose(g_norms,0)
    vaild_idx=np.nonzero(vaild_mask)[0]
    vaild_sim=sim[vaild_mask]
    if not isinstance(k,int)or k<=0 or k>len(vaild_sim):
        raise ValueError("k必须是正整数且不大于有效向量个数")
    sort_pos=np.argsort(-vaild_sim,kind='stable')
    topk_pos=sort_pos[:k]
    topk_idx=vaild_idx[topk_pos]
    topk_sim=vaild_sim[topk_pos]
    return topk_idx,topk_sim
if __name__=="__main__":
    query = np.array([0.60, -0.80, 0.00])
    gallery = np.array([
        [0.60, -0.80, 0.00],
        [0.30, -0.40, 0.00],
        [0.80, 0.60, 0.00],
        [-0.60, 0.80, 0.00],
        [0.00, 0.00, 0.00]
    ])
    ids=np.array(["img_a","img_b","img_c","img_d","img_zero"])
    k=3
    idx,sim=top_k_cosine(query,gallery,k)
    print("\nM2 top-k索引",idx)
    print("M2相似度",sim)
    print("选中的图像ID",ids[idx])

#11
def pairwise_dist(A,B):
    diff=A[:,np.newaxis,:]-B[np.newaxis,:,:]
    dist_sq=np.sum(diff**2,axis=-1)
    D=np.sqrt(dist_sq)
    nearest_idx=np.argmin(D,axis=1)
    return D,nearest_idx

if __name__ == "__main__":
    A = np.array([
        [0.20, -0.40],
        [0.80, 0.30],
        [-0.50, 0.70]
    ])
    B = np.array([
        [0.10, 0.20],
        [-0.60, -0.20]
    ])
    D, nearest_idx = pairwise_dist(A, B)
    print("==== NP-M3 ====")
    print("距离矩阵 D:\n", D)
    print("D shape:", D.shape)
    print("A各行距离最近的B索引：", nearest_idx)


#12
def forward(X, W1, b1, W2, b2):
    
    Z = X @ W1.T + b1
    # H = relu
    H = np.maximum(Z, 0.0)
    # Y = H @ W2.T + b2
    Y = H @ W2.T + b2
    return Z, H, Y

if __name__ == "__main__":
    X = np.array([
        [0.80, -0.60, 0.70],
        [-0.40, -0.50, 0.90]
    ])
    W1 = np.array([
        [-0.10, 0.20, -0.80],
        [0.30, -0.50, 0.10]
    ])
    b1 = np.array([-0.15, 0.05])
    W2 = np.array([
        [0.20, -0.60],
        [-0.80, 0.52]
    ])
    b2 = np.array([0.10, -0.20])

    Z, H, Y = forward(X, W1, b1, W2, b2)
    print("\n==== NP-M4 ====")
    print("Z:\n", Z)
    print("Z shape:", Z.shape)
    print("H:\n", H)
    print("H shape:", H.shape)
    print("Y:\n", Y)
    print("Y shape:", Y.shape)
