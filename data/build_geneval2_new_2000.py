"""生成可配置规模的 geneval2 风格 instruction。

模板照官方 8 类：single_object / two_object / counting / colors / color_attr /
position / verb + 已生成 400 条里出现过的更复杂模板（多 slot 嵌套）。

限制：
- 与官方 raw 800 条不能出现连续 3 个相同的词（无视大小写、单复数）
- 与已生成 400 条同样限制（做为参考）
- 与额外指定的历史 construction pool 做同样的 3-gram 去重
- 新批次内部也做 3-gram 去重，并按 atom_count 3--10 均衡采样

产出：
- geneval2_new_2000.jsonl（GenEval2 官方格式：prompt / atom_count / vqa_list / skills）
- geneval2_qwen_edit_gen2000_perseed.json（本项目格式：case_id / instruction / scores）
- 同名 flux 版本
"""
import argparse
import json, random, os, re, tempfile
from collections import Counter


parser = argparse.ArgumentParser(
    description="Generate a disjoint, atom-balanced GenEval2 construction pool."
)
parser.add_argument("--target", type=int, default=2000)
parser.add_argument("--seed", type=int, default=20260727)
parser.add_argument(
    "--output",
    default=None,
    help="Output JSONL (default: data/geneval2_new_2000.jsonl).",
)
parser.add_argument(
    "--blacklist",
    action="append",
    default=[],
    help="Additional JSONL whose prompts must be excluded by normalized 3-gram.",
)
parser.add_argument(
    "--tag",
    default="gen2000",
    help="Tag used by optional empty per-seed placeholder files.",
)
parser.add_argument(
    "--skip-placeholders",
    action="store_true",
    help="Do not write empty score placeholder JSON files.",
)
ARGS = parser.parse_args()

if ARGS.target <= 0:
    parser.error("--target must be positive")

random.seed(ARGS.seed)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_JSONL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/UMM/Benchmark/GenEval2-main/geneval2_data.jsonl"
GEN400_JSONL = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhaobo45/yihang/26AAAIworks/codes/geneval2/generated_geneval2_data_400.jsonl"

OUT_DIR = os.path.join(HERE, "final_data/geneval2/edit")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSONL = ARGS.output or os.path.join(HERE, "geneval2_new_2000.jsonl")

TARGET = ARGS.target

# ── 词表（照官方 + 已生成扩展）────────────────────────────────────────────────
COUNT_WORDS = ["two", "three", "four", "five", "six", "seven"]  # single 用 "a"
COUNT_TO_NUM = {"two":2, "three":3, "four":4, "five":5, "six":6, "seven":7}

COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown",
          "purple", "pink", "azure", "crimson", "emerald", "violet",
          "rose", "ebony", "ivory", "tan", "orange", "gray", "cyan",
          "maroon", "teal", "navy", "silver", "scarlet", "turquoise",
          "indigo", "lavender", "beige", "coral", "amber", "magenta"]
MATERIALS = ["wooden", "metal", "metallic", "plastic", "stone", "rocky",
             "glass", "crystal", "golden", "synthetic", "timber",
             "ceramic", "paper", "leather", "rubber", "concrete",
             "bronze", "copper", "marble", "porcelain", "fabric",
             "cardboard", "wicker", "clay"]
PATTERNS = ["striped", "spotted", "checkered", "dotted", "plaid", "lined",
            "sparkling", "glittering", "speckled", "mottled", "marbled",
            "woven", "floral", "geometric", "zigzag", "gradient"]
ALL_ATTRS = COLORS + MATERIALS + PATTERNS

OBJECTS = ["backpacks","bagels","bears","bicycles","birds","candles","cars",
           "cats","chairs","clocks","cookies","cows","croissants","dogs",
           "donuts","elephants","flamingos","flowers","giraffes","guitars",
           "horses","kangaroos","koalas","lions","monkeys","motorcycles",
           "mushrooms","penguins","pigs","rabbits","raccoons","sheeps",
           "suitcases","toys","trucks","trumpets","turtles","umbrellas",
           "violins","zebras",
           "airplanes","apples","balloons","bananas","benches","boats",
           "books","bottles","bowls","buses","cameras","canoes","carts",
           "castles","cheeses","chickens","cups","deer","desks","dolphins",
           "doors","drums","ducks","eagles","fans","flags","forks","foxes",
           "frogs","glasses","goats","hammers","hats","helmets","kettles",
           "keys","lamps","lemons","lizards","magnets","mice","mirrors",
           "moons","notebooks","oranges","owls","paintings","peaches","pears",
           "pianos","plates","robots","rockets","sailboats","sharks","shoes",
           "skateboards","snakes","spoons","stars","stools","tables","teapots",
           "telephones","tents","tomatoes","tractors","trains","trees","vases",
           "wagons","watches","watermelons","whales","windows","wolves"]

# 单数形（官方词表里全部是 -s 结尾复数，直接去 s）
_SING_TABLE = {
    "sheeps": "sheep", "benches": "bench", "buses": "bus",
    "cheeses": "cheese", "foxes": "fox", "glasses": "glass",
    "mice": "mouse", "peaches": "peach", "watches": "watch",
    "wolves": "wolf",
}
def sing(pl):
    if pl in _SING_TABLE: return _SING_TABLE[pl]
    if pl.endswith("s"): return pl[:-1]
    return pl

POSITIONS = ["above", "below", "behind", "in front of", "under", "on top of",
             "to the left of", "to the right of", "on the left side of",
             "on the right side of", "ahead of", "in back of", "beneath"]
VERBS = ["chasing", "jumping over", "leaping over", "playing with",
         "interacting with", "pursuing"]

# ── 3-gram 归一化 ────────────────────────────────────────────────────────────
# 简单规则：小写 + 去掉词尾单个 's'（sheep/sheeps 特殊映射）。
_word_re = re.compile(r"[A-Za-z]+")
def _norm_word(w):
    w = w.lower()
    if w in ("sheep", "sheeps"): return "sheep"
    if len(w) > 1 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w

def trigrams(text):
    ws = [_norm_word(w) for w in _word_re.findall(text)]
    return {tuple(ws[i:i+3]) for i in range(len(ws)-2)}

# ── 装基准数据和额外 construction pool 的 3-gram 黑名单 ─────────────────────
def load_prompts(path, key="prompt"):
    return [json.loads(l)[key] for l in open(path)]

BAD_TRI = set()
blacklist_paths = [RAW_JSONL, GEN400_JSONL] + ARGS.blacklist
blacklist_prompts = []
for path in blacklist_paths:
    blacklist_prompts.extend(load_prompts(path))
for prompt in blacklist_prompts:
    BAD_TRI |= trigrams(prompt)
print(
    f"[黑名单] {len(blacklist_prompts)} prompts / "
    f"{len(BAD_TRI)} normalized 3-grams from {len(blacklist_paths)} files",
    flush=True,
)

# ── 生成模板 ─────────────────────────────────────────────────────────────────
def pick(pool, exclude=()):
    while True:
        x = random.choice(pool)
        if x not in exclude: return x

def obj_phrase(count, attrs, obj_pl):
    """count: 'a' or count word；attrs: list of adjectives；obj_pl: 复数形"""
    if count == "a":
        head = f"a {' '.join(attrs)} {sing(obj_pl)}" if attrs else f"a {sing(obj_pl)}"
    else:
        head = f"{count} {' '.join(attrs)} {obj_pl}" if attrs else f"{count} {obj_pl}"
    return head.strip()

def build_vqa(entities, relations):
    """entities: [{'count','attrs','obj_pl','singular_ref':bool}]
    relations: [(i, position/verb, j)]
    生成 vqa_list + skills + atom_count，格式对齐 geneval2_data.jsonl。

    atom_count 规则（照官方 README）：a/one 的 count 不计入，其他 skill 全部计入。
    """
    vqa, skills, atom = [], [], 0
    for k, e in enumerate(entities):
        obj_sing_or_pl = sing(e['obj_pl']) if e['singular_ref'] else e['obj_pl']
        num_word = "one" if e['count'] == "a" else e['count']
        vqa.append((f"How many {e['obj_pl']} are in the image?", num_word))
        skills.append("count")
        if e['count'] != "a":  # a/one 的 count 不算
            atom += 1
        vqa.append((f"Are there any {e['obj_pl']} in the image?", "Yes"))
        skills.append("object"); atom += 1
        for a in e['attrs']:
            article_verb = "Is" if e['singular_ref'] else "Are"
            vqa.append((f"{article_verb} the {obj_sing_or_pl} {a}?", "Yes"))
            skills.append("attribute"); atom += 1
        for (src, rel, tgt) in relations:
            if src != k: continue
            tgt_e = entities[tgt]
            tgt_ref = sing(tgt_e['obj_pl']) if tgt_e['singular_ref'] else tgt_e['obj_pl']
            article_verb = "Is" if e['singular_ref'] else "Are"
            vqa.append((f"{article_verb} the {obj_sing_or_pl} {rel} the {tgt_ref}?", "Yes"))
            skills.append("position" if rel in POSITIONS else "verb"); atom += 1
    return vqa, skills, atom

def make_single_object():
    obj = random.choice(OBJECTS)
    is_single = random.random() < 0.5
    count = "a" if is_single else random.choice(COUNT_WORDS)
    prompt = obj_phrase(count, [], obj)
    ent = [dict(count=count, attrs=[], obj_pl=obj, singular_ref=(count=="a"))]
    vqa, sk, ac = build_vqa(ent, [])
    return prompt, ac, vqa, sk

def make_attributed_object(n_attrs=1):
    """带 n_attrs 个属性（颜色/材料/纹样）的单实体，count 随机。"""
    obj = random.choice(OBJECTS)
    is_single = random.random() < 0.5
    count = "a" if is_single else random.choice(COUNT_WORDS)
    attrs = random.sample(ALL_ATTRS, n_attrs)
    prompt = obj_phrase(count, attrs, obj)
    ent = [dict(count=count, attrs=attrs, obj_pl=obj, singular_ref=(count=="a"))]
    vqa, sk, ac = build_vqa(ent, [])
    return prompt, ac, vqa, sk

def make_two_object():
    """两实体 and 连接，各自可带属性。"""
    o1, o2 = random.sample(OBJECTS, 2)
    def _side():
        is_single = random.random() < 0.5
        count = "a" if is_single else random.choice(COUNT_WORDS)
        attrs = random.sample(ALL_ATTRS, random.choice([0,0,1,1,2])) if random.random() < 0.7 else []
        return count, attrs
    c1, a1 = _side(); c2, a2 = _side()
    p1 = obj_phrase(c1, a1, o1); p2 = obj_phrase(c2, a2, o2)
    prompt = f"{p1} and {p2}"
    ents = [dict(count=c1, attrs=a1, obj_pl=o1, singular_ref=(c1=="a")),
            dict(count=c2, attrs=a2, obj_pl=o2, singular_ref=(c2=="a"))]
    vqa, sk, ac = build_vqa(ents, [])
    return prompt, ac, vqa, sk

def make_position():
    o1, o2 = random.sample(OBJECTS, 2)
    rel = random.choice(POSITIONS)
    def _side():
        is_single = random.random() < 0.4
        count = "a" if is_single else random.choice(COUNT_WORDS)
        attrs = random.sample(ALL_ATTRS, random.choice([0,1,1,2])) if random.random() < 0.7 else []
        return count, attrs
    c1, a1 = _side(); c2, a2 = _side()
    p1 = obj_phrase(c1, a1, o1); p2 = obj_phrase(c2, a2, o2)
    prompt = f"{p1} {rel} {p2}"
    ents = [dict(count=c1, attrs=a1, obj_pl=o1, singular_ref=(c1=="a")),
            dict(count=c2, attrs=a2, obj_pl=o2, singular_ref=(c2=="a"))]
    vqa, sk, ac = build_vqa(ents, [(0, rel, 1)])
    return prompt, ac, vqa, sk

def make_verb():
    o1, o2 = random.sample(OBJECTS, 2)
    verb = random.choice(VERBS)
    def _side(single_bias=0.5):
        is_single = random.random() < single_bias
        count = "a" if is_single else random.choice(COUNT_WORDS)
        attrs = random.sample(ALL_ATTRS, random.choice([0,1])) if random.random() < 0.5 else []
        return count, attrs
    c1, a1 = _side(0.5); c2, a2 = _side(0.5)
    p1 = obj_phrase(c1, a1, o1); p2 = obj_phrase(c2, a2, o2)
    prompt = f"{p1} {verb} {p2}"
    ents = [dict(count=c1, attrs=a1, obj_pl=o1, singular_ref=(c1=="a")),
            dict(count=c2, attrs=a2, obj_pl=o2, singular_ref=(c2=="a"))]
    vqa, sk, ac = build_vqa(ents, [(0, verb, 1)])
    return prompt, ac, vqa, sk

def make_triple_and():
    """三实体并列：'A, B and C'。"""
    o1, o2, o3 = random.sample(OBJECTS, 3)
    def _side():
        is_single = random.random() < 0.4
        count = "a" if is_single else random.choice(COUNT_WORDS)
        attrs = random.sample(ALL_ATTRS, random.choice([0,1,1,2])) if random.random() < 0.7 else []
        return count, attrs
    c1,a1 = _side(); c2,a2 = _side(); c3,a3 = _side()
    p1 = obj_phrase(c1,a1,o1); p2 = obj_phrase(c2,a2,o2); p3 = obj_phrase(c3,a3,o3)
    # 两种连接：'A, B and C' 或 'A, B, and C'
    prompt = f"{p1}, {p2} and {p3}"
    ents = [dict(count=c1,attrs=a1,obj_pl=o1,singular_ref=(c1=="a")),
            dict(count=c2,attrs=a2,obj_pl=o2,singular_ref=(c2=="a")),
            dict(count=c3,attrs=a3,obj_pl=o3,singular_ref=(c3=="a"))]
    vqa,sk,ac = build_vqa(ents, [])
    return prompt, ac, vqa, sk

def make_position_chain():
    """A rel1 B rel2 C。"""
    o1, o2, o3 = random.sample(OBJECTS, 3)
    r1, r2 = random.choice(POSITIONS), random.choice(POSITIONS)
    def _side():
        is_single = random.random() < 0.4
        count = "a" if is_single else random.choice(COUNT_WORDS)
        attrs = random.sample(ALL_ATTRS, random.choice([0,1])) if random.random() < 0.6 else []
        return count, attrs
    c1,a1 = _side(); c2,a2 = _side(); c3,a3 = _side()
    p1 = obj_phrase(c1,a1,o1); p2 = obj_phrase(c2,a2,o2); p3 = obj_phrase(c3,a3,o3)
    prompt = f"{p1} {r1} {p2} {r2} {p3}"
    ents = [dict(count=c1,attrs=a1,obj_pl=o1,singular_ref=(c1=="a")),
            dict(count=c2,attrs=a2,obj_pl=o2,singular_ref=(c2=="a")),
            dict(count=c3,attrs=a3,obj_pl=o3,singular_ref=(c3=="a"))]
    vqa,sk,ac = build_vqa(ents, [(0,r1,1),(1,r2,2)])
    return prompt, ac, vqa, sk

def make_verb_plus_and():
    """A verb B, and C。"""
    o1, o2, o3 = random.sample(OBJECTS, 3)
    v = random.choice(VERBS)
    def _side():
        is_single = random.random() < 0.4
        count = "a" if is_single else random.choice(COUNT_WORDS)
        attrs = random.sample(ALL_ATTRS, random.choice([0,1])) if random.random() < 0.6 else []
        return count, attrs
    c1,a1 = _side(); c2,a2 = _side(); c3,a3 = _side()
    p1 = obj_phrase(c1,a1,o1); p2 = obj_phrase(c2,a2,o2); p3 = obj_phrase(c3,a3,o3)
    prompt = f"{p1} {v} {p2}, and {p3}"
    ents = [dict(count=c1,attrs=a1,obj_pl=o1,singular_ref=(c1=="a")),
            dict(count=c2,attrs=a2,obj_pl=o2,singular_ref=(c2=="a")),
            dict(count=c3,attrs=a3,obj_pl=o3,singular_ref=(c3=="a"))]
    vqa,sk,ac = build_vqa(ents, [(0,v,1)])
    return prompt, ac, vqa, sk

def make_dense_pos_chain():
    """高密度：3 实体位置链，每实体强制 count 非 a，1-2 attr。atom≈9-13。"""
    o1, o2, o3 = random.sample(OBJECTS, 3)
    r1, r2 = random.choice(POSITIONS), random.choice(POSITIONS)
    def _side():
        count = random.choice(COUNT_WORDS)
        na = random.choice([1, 1, 2])
        attrs = random.sample(ALL_ATTRS, na)
        return count, attrs
    c1,a1 = _side(); c2,a2 = _side(); c3,a3 = _side()
    p1 = obj_phrase(c1,a1,o1); p2 = obj_phrase(c2,a2,o2); p3 = obj_phrase(c3,a3,o3)
    prompt = f"{p1} {r1} {p2} {r2} {p3}"
    ents = [dict(count=c1,attrs=a1,obj_pl=o1,singular_ref=False),
            dict(count=c2,attrs=a2,obj_pl=o2,singular_ref=False),
            dict(count=c3,attrs=a3,obj_pl=o3,singular_ref=False)]
    vqa,sk,ac = build_vqa(ents, [(0,r1,1),(1,r2,2)])
    return prompt, ac, vqa, sk

def make_dense_verb_and_pos():
    """高密度：'A verb B rel C' — 3 实体，每个强 count + 1-2 attr。atom≈10-12。"""
    o1, o2, o3 = random.sample(OBJECTS, 3)
    v = random.choice(VERBS); r = random.choice(POSITIONS)
    def _side():
        count = random.choice(COUNT_WORDS)
        attrs = random.sample(ALL_ATTRS, random.choice([1, 2]))
        return count, attrs
    c1,a1 = _side(); c2,a2 = _side(); c3,a3 = _side()
    p1 = obj_phrase(c1,a1,o1); p2 = obj_phrase(c2,a2,o2); p3 = obj_phrase(c3,a3,o3)
    prompt = f"{p1} {v} {p2} {r} {p3}"
    ents = [dict(count=c1,attrs=a1,obj_pl=o1,singular_ref=False),
            dict(count=c2,attrs=a2,obj_pl=o2,singular_ref=False),
            dict(count=c3,attrs=a3,obj_pl=o3,singular_ref=False)]
    vqa,sk,ac = build_vqa(ents, [(0,v,1),(1,r,2)])
    return prompt, ac, vqa, sk

def make_dense_two_object():
    """高密度：2 实体，各 2 attr + count 强制，有关系。atom≈9。"""
    o1, o2 = random.sample(OBJECTS, 2)
    rel = random.choice(POSITIONS + VERBS)
    c1 = random.choice(COUNT_WORDS); c2 = random.choice(COUNT_WORDS)
    a1 = random.sample(ALL_ATTRS, 2); a2 = random.sample(ALL_ATTRS, 2)
    p1 = obj_phrase(c1, a1, o1); p2 = obj_phrase(c2, a2, o2)
    prompt = f"{p1} {rel} {p2}"
    ents = [dict(count=c1,attrs=a1,obj_pl=o1,singular_ref=False),
            dict(count=c2,attrs=a2,obj_pl=o2,singular_ref=False)]
    vqa,sk,ac = build_vqa(ents, [(0,rel,1)])
    return prompt, ac, vqa, sk

# 分布权重（简单实体多一些，复杂实体少一些，与官方 raw 分布近似）
TEMPLATES = [
    (make_single_object,       5),
    (make_attributed_object,   10),  # 1 attr
    (lambda: make_attributed_object(2), 4),
    (make_two_object,          16),
    (make_position,            14),
    (make_verb,                8),
    (make_triple_and,          10),
    (make_position_chain,      8),
    (make_verb_plus_and,       6),
    (make_dense_two_object,    8),
    (make_dense_pos_chain,     10),
    (make_dense_verb_and_pos,  10),
]
WEIGHTS = []
POOL = []
for fn, w in TEMPLATES:
    POOL.append(fn); WEIGHTS.append(w)

def sample_prompt():
    fn = random.choices(POOL, weights=WEIGHTS, k=1)[0]
    return fn()

# ── 精确按 atom_count 生成 ────────────────────────────────────────────────
# atom = n_ent + n_ent_plural + n_attrs_total + n_relations
#   （count 只在非 "a" 时算，object 每实体算 1，attribute 每个算 1，rel 每个算 1）
def make_by_atom(target_ac):
    configs = []
    for n_ent in [1, 2, 3]:
        rel_opts = [0] if n_ent == 1 else [0, n_ent - 1]
        for n_rel in rel_opts:
            base = n_ent + n_rel
            rem = target_ac - base
            if rem < 0: continue
            for n_plural in range(0, n_ent + 1):
                n_attrs = rem - n_plural
                if 0 <= n_attrs <= n_ent * 3:
                    configs.append((n_ent, n_rel, n_plural, n_attrs))
    if not configs: return None
    n_ent, n_rel, n_plural, n_attrs = random.choice(configs)
    # 分配 attrs
    attr_dist = [0] * n_ent
    for _ in range(n_attrs):
        cands = [i for i in range(n_ent) if attr_dist[i] < 3]
        if not cands: break
        attr_dist[random.choice(cands)] += 1
    # 分配 plural
    plural = [False] * n_ent
    for i in random.sample(range(n_ent), n_plural):
        plural[i] = True
    # 组装
    objs = random.sample(OBJECTS, n_ent)
    entities = []
    for k in range(n_ent):
        count = random.choice(COUNT_WORDS) if plural[k] else "a"
        attrs = random.sample(ALL_ATTRS, attr_dist[k])
        entities.append(dict(count=count, attrs=attrs, obj_pl=objs[k],
                             singular_ref=(count == "a")))
    if n_rel == 0:
        parts = [obj_phrase(e['count'], e['attrs'], e['obj_pl']) for e in entities]
        if n_ent == 1: prompt = parts[0]
        elif n_ent == 2: prompt = f"{parts[0]} and {parts[1]}"
        else: prompt = f"{parts[0]}, {parts[1]} and {parts[2]}"
        rels = []
    else:
        parts = [obj_phrase(e['count'], e['attrs'], e['obj_pl']) for e in entities]
        if n_ent == 2:
            rel = random.choice(POSITIONS + VERBS)
            prompt = f"{parts[0]} {rel} {parts[1]}"
            rels = [(0, rel, 1)]
        else:
            r1 = random.choice(POSITIONS + VERBS)
            r2 = random.choice(POSITIONS + VERBS)
            prompt = f"{parts[0]} {r1} {parts[1]} {r2} {parts[2]}"
            rels = [(0, r1, 1), (1, r2, 2)]
    vqa, sk, ac = build_vqa(entities, rels)
    return prompt, ac, vqa, sk

# ── 主循环：按 atom_count 分档抽样 ────────────────────────────────────────
# 官方 raw 800 是 atom_count 3-10 各 100 条。任意 target 都尽量均匀分档。
bucket_base, bucket_remainder = divmod(TARGET, 8)
TARGET_BUCKETS = {
    ac: bucket_base + int(ac - 3 < bucket_remainder)
    for ac in range(3, 11)
}
buckets = {ac: [] for ac in TARGET_BUCKETS}

kept_tri = set()
seen_prompt = set()
n_tries = 0
n_out_of_range = 0  # atom_count 不在 3-10
n_bucket_full = 0
n_dup = 0
n_ngram_bad = 0
n_self_dup = 0

def done():
    return all(len(buckets[ac]) >= TARGET_BUCKETS[ac] for ac in buckets)

def pick_missing_ac():
    """按缺口大小加权抽 atom_count。"""
    gaps = [(ac, TARGET_BUCKETS[ac] - len(buckets[ac])) for ac in buckets]
    gaps = [(ac, g) for ac, g in gaps if g > 0]
    if not gaps: return None
    acs, ws = zip(*gaps)
    return random.choices(acs, weights=ws, k=1)[0]

MAX_TRIES = TARGET * 200
while not done():
    n_tries += 1
    if n_tries > MAX_TRIES:
        print(f"[warn] 尝试 {n_tries} 次，仍未装满，提前停")
        break
    # 缺哪档就精确造哪档
    target_ac = pick_missing_ac()
    if target_ac is None: break
    try:
        r = make_by_atom(target_ac)
        if r is None: continue
        prompt, ac, vqa, sk = r
    except Exception:
        continue
    if ac not in TARGET_BUCKETS:
        n_out_of_range += 1; continue
    if len(buckets[ac]) >= TARGET_BUCKETS[ac]:
        n_bucket_full += 1; continue
    key = prompt.lower()
    if key in seen_prompt:
        n_dup += 1; continue
    tri = trigrams(prompt)
    if tri & BAD_TRI:
        n_ngram_bad += 1; continue
    if tri & kept_tri:
        n_self_dup += 1; continue
    seen_prompt.add(key)
    kept_tri |= tri
    buckets[ac].append(dict(prompt=prompt, atom_count=ac, vqa_list=vqa, skills=sk))
    total = sum(len(v) for v in buckets.values())
    if total % 200 == 0:
        counts = " ".join(f"{ac}:{len(buckets[ac])}" for ac in range(3, 11))
        print(f"  收到 {total}  {counts}  raw冲突={n_ngram_bad} 内部冲突={n_self_dup} 重复={n_dup} 满桶={n_bucket_full}", flush=True)

if not done():
    counts = {ac: len(buckets[ac]) for ac in sorted(buckets)}
    raise RuntimeError(
        f"只生成 {sum(counts.values())}/{TARGET} 条，拒绝写出不完整数据: {counts}"
    )

kept = []
for ac in range(3, 11):
    kept.extend(buckets[ac])

print(f"\n最终: {len(kept)} 条  总尝试={n_tries}")
print(f"  官方/前400 3-gram 冲突丢弃: {n_ngram_bad}")
print(f"  内部 3-gram 冲突丢弃:      {n_self_dup}")
print(f"  完全同 prompt 丢弃:        {n_dup}")
print(f"  atom_count 越界丢弃:       {n_out_of_range}")
print(f"  桶满丢弃:                  {n_bucket_full}")

# 模板分布统计
temp_counts = Counter()
for k in kept:
    temp_counts[k['atom_count']] += 1
print("atom_count 分布:", sorted(temp_counts.items()))

# 写 GenEval2 官方风格 jsonl。先写同目录临时文件，避免任务中断留下截断 JSONL。
out_parent = os.path.dirname(os.path.abspath(OUT_JSONL))
os.makedirs(out_parent, exist_ok=True)
tmp_path = None
try:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=out_parent,
        prefix=os.path.basename(OUT_JSONL) + ".", suffix=".tmp",
        delete=False,
    ) as f:
        tmp_path = f.name
        for x in kept:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, OUT_JSONL)
    tmp_path = None
finally:
    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)
print(f"[写出] {OUT_JSONL}  ({len(kept)} 条)")

# 写本项目 gen_perseed 风格 json（case_id + instruction + scores 占位空 dict）
def write_perseed(fp, kept):
    records = []
    for i, x in enumerate(kept):
        records.append({
            "case_id": str(i + 1),
            "instruction": x['prompt'],
            "scores": {},
        })
    parent = os.path.dirname(os.path.abspath(fp))
    os.makedirs(parent, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=parent,
            prefix=os.path.basename(fp) + ".", suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = f.name
            json.dump(records, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, fp)
        tmp_path = None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    print(f"[写出] {fp}  ({len(records)} 条)")

if not ARGS.skip_placeholders:
    write_perseed(
        os.path.join(OUT_DIR, f"geneval2_qwen_edit_{ARGS.tag}_perseed.json"),
        kept,
    )
    write_perseed(
        os.path.join(OUT_DIR, f"geneval2_flux_edit_{ARGS.tag}_perseed.json"),
        kept,
    )

print("\n样例:")
for x in kept[:10]:
    print(" -", x['prompt'])
