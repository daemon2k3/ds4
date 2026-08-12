import struct, sys, collections

P = sys.argv[1]
f = open(P, "rb")
def rd(fmt):
    b = f.read(struct.calcsize(fmt)); return struct.unpack(fmt, b)
def rstr():
    (n,) = rd("<Q"); return f.read(n).decode("utf-8", "replace")
GT = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",7:"bool",8:"str",9:"arr",10:"u64",11:"i64",12:"f64"}
QT = {0:("F32",4,1),1:("F16",2,1),2:("Q4_0",None,32),3:("Q4_1",None,32),6:("Q5_0",None,32),7:("Q5_1",None,32),8:("Q8_0",34,32),9:("Q8_1",None,32),10:("Q2_K",None,256),11:("Q3_K",None,256),12:("Q4_K",None,256),13:("Q5_K",None,256),14:("Q6_K",None,256),15:("Q8_K",None,256),16:("IQ2_XXS",None,256),17:("IQ2_XS",None,256),18:("IQ3_XXS",None,256),19:("IQ1_S",None,256),20:("IQ4_NL",None,256),39:("MXFP4",None,32)}
# approximate bytes/block for known quants (block_elems, bytes/block)
QBYTES = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),10:(256,68),11:(256,110),12:(256,144),13:(256,176),14:(256,210),15:(256,292),16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(256,136),39:(32,17)}

magic, = rd("<I"); assert magic == 0x46554747, hex(magic)
ver, = rd("<I"); ntensors, = rd("<Q"); nkv, = rd("<Q")
print(f"GGUF v{ver} tensors={ntensors} metadata={nkv}")
meta = {}
for _ki in range(nkv):
    _off = f.tell()
    k = rstr(); (t,) = rd("<I")
    _last = (k, t, _off)
    if t == 9:
        (et,) = rd("<I"); (n,) = rd("<Q")
        vals = []
        for _ in range(min(n, 8)):
            if GT[et]=="str": vals.append(rstr())
            elif GT[et] in ("u8","i8","bool"): vals.append(rd("<B")[0])
            elif GT[et]=="u16": vals.append(rd("<H")[0])
            elif GT[et]=="i16": vals.append(rd("<h")[0])
            elif GT[et]=="u32": vals.append(rd("<I")[0])
            elif GT[et]=="i32": vals.append(rd("<i")[0])
            elif GT[et]=="u64": vals.append(rd("<Q")[0])
            elif GT[et]=="i64": vals.append(rd("<q")[0])
            elif GT[et]=="f32": vals.append(rd("<f")[0])
            elif GT[et]=="f64": vals.append(rd("<d")[0])
        if n > 8:  # skip rest
            if GT[et] == "str":
                for _ in range(n-8): rstr()
            else:
                sz = {"u8":1,"i8":1,"bool":1,"u16":2,"i16":2,"u32":4,"i32":4,"f32":4,"u64":8,"i64":8,"f64":8}[GT[et]]
                f.seek((n-8)*sz, 1)
        meta[k] = (GT[et], n, vals)
    elif t == 8: meta[k] = rstr()
    elif t in (0,1,7): meta[k] = rd("<B")[0]
    elif t == 2: meta[k] = rd("<H")[0]
    elif t == 3: meta[k] = rd("<h")[0]
    elif t == 4: meta[k] = rd("<I")[0]
    elif t == 5: meta[k] = rd("<i")[0]
    elif t == 10: meta[k] = rd("<Q")[0]
    elif t == 11: meta[k] = rd("<q")[0]
    elif t == 6: meta[k] = rd("<f")[0]
    elif t == 12: meta[k] = rd("<d")[0]

interesting = {k:v for k,v in meta.items() if any(s in k for s in ("general","deepseek","expert","attention","context","block_count","embedding_length","feed_forward","rope","chat","template","token","moe","head","compress","index","hc","scale","type","quant","name","author","version","license","file_type"))}
for k in sorted(interesting):
    v = meta[k]
    s = str(v)
    print(f"  {k} = {s[:160]}")

def tclass(name):
    n = name.lower()
    for pat, cls in [("token_embd","embd"),("output_norm","norm"),("output.weight","out"),("exp_probs_b","router-bias"),("gate.weight","router"),("router","router"),
                     ("index","indexer"),("compress","compressor"),("hc","hc"),("attn_q_a","attn"),("attn_q_b","attn"),("attn_kv_a","attn"),("attn_kv_b","attn"),
                     ("attn","attn"),("q_a_norm","attn"),("kv_a_norm","attn"),("tid2eid","lookup"),
                     ("ffn_gate_exps","routed-gate"),("ffn_up_exps","routed-up"),("ffn_down_exps","routed-down"),
                     ("ffn_gate_shexp","shared-gate"),("ffn_up_shexp","shared-up"),("ffn_down_shexp","shared-down"),
                     ("ffn","ffn"),("shexp","shared"),("exps","routed"),("_norm","norm")]:
        if pat in n: return cls
    return "other"

infos = []
data_off_end = 0
for _ in range(ntensors):
    name = rstr()
    (nd,) = rd("<I"); dims = rd(f"<{nd}Q"); (qt,) = rd("<I"); (off,) = rd("<Q")
    elems = 1
    for d in dims: elems *= d
    infos.append((name, elems, qt, off))

ALIGN = meta.get("general.alignment", 32)
data_base = (f.tell() + ALIGN - 1) // ALIGN * ALIGN
QNAMES = {0:"F32",1:"F16",2:"Q4_0",8:"Q8_0",10:"Q2_K",12:"Q4_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",39:"MXFP4",26:"qt26",17:"IQ2_XS"}

census = collections.defaultdict(lambda: [0,0,0])
layers = set()
for i,(name, elems, qt, off) in enumerate(infos):
    nxt = infos[i+1][3] if i+1 < len(infos) else None
    nbytes = (nxt - off) if nxt is not None else None
    census[(tclass(name), qt)][0] += 1
    census[(tclass(name), qt)][1] += elems
    if nbytes is not None: census[(tclass(name), qt)][2] += nbytes
    if name.startswith("blk."): layers.add(int(name.split(".")[1]))

print(f"\n== census (class / raw qtype: tensors, params, GiB-from-offsets, bits/param) ==")
tot_e = tot_b = 0
for (cls, qt), (c, e, b) in sorted(census.items(), key=lambda x: -x[1][2]):
    tot_e += e; tot_b += b
    print(f"  {cls:14s} qt={qt:2d} n={c:5d} params={e/1e9:7.3f}B  {b/2**30:8.2f} GiB  ({b/e*8:.2f} b/p)")
print(f"  TOTAL          n={ntensors:5d} params={tot_e/1e9:7.2f}B  {tot_b/2**30:8.2f} GiB ({tot_b/tot_e*8:.2f} b/p)")
print(f"\nlayers: {len(layers)} [{min(layers)}..{max(layers)}], data_base={data_base}")
# raw type histogram for unknown codes
tcnt = collections.Counter(qt for _,_,qt,_ in infos)
print("raw qtype histogram:", dict(sorted(tcnt.items())))
