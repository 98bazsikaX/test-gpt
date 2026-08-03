import math
import random as _random

import numpy as np

import microgpt


def _rmsnorm1(x):
    ms = (x * x).mean() + 1e-5
    return x / np.sqrt(ms)


def _softmax1(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def reference_forward(inp, sd):
    """Naive per-position forward, mirroring the original scalar microgpt exactly."""
    n_layer = microgpt.n_layer
    n_head = microgpt.n_head
    head_dim = microgpt.head_dim
    n_embd = microgpt.n_embd
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    logits_out = []
    for pos, tok in enumerate(inp):
        x = sd["token_embedding"][tok] + sd["position_embedding"][pos]
        x = _rmsnorm1(x)
        for li in range(n_layer):
            x_res = x
            x = _rmsnorm1(x)
            q = x @ sd[f"layer{li}.query_w"].T
            k = x @ sd[f"layer{li}.key_w"].T
            v = x @ sd[f"layer{li}.value_w"].T
            keys[li].append(k)
            values[li].append(v)
            Tcur = len(keys[li])
            K = np.stack(keys[li])
            V = np.stack(values[li])
            attn = np.zeros(n_embd)
            for h in range(n_head):
                hs = h * head_dim
                qh = q[hs : hs + head_dim]
                kh = K[:, hs : hs + head_dim]
                vh = V[:, hs : hs + head_dim]
                scores = np.array([qh @ kh[t] for t in range(Tcur)]) / head_dim ** 0.5
                w = _softmax1(scores)
                attn[hs : hs + head_dim] = sum(w[t] * vh[t] for t in range(Tcur))
            x = (attn @ sd[f"layer{li}.output_w"].T) + x_res
            x_res = x
            x = _rmsnorm1(x)
            x = np.maximum(0, x @ sd[f"layer{li}.fc1_w"].T)
            x = (x @ sd[f"layer{li}.fc2_w"].T) + x_res
        logits_out.append(x @ sd["lm_head"].T)
    return np.stack(logits_out)


def test_batched_forward_matches_per_position_reference():
    np.random.seed(42)
    microgpt.reset_params()
    inp = np.array([microgpt.BOS, 1, 3, 7, microgpt.BOS], dtype=np.int64)
    logits, _ = microgpt.forward(inp)
    ref = reference_forward(inp, microgpt.state_dict)
    np.testing.assert_allclose(logits, ref, rtol=1e-5, atol=1e-5)


class _ReferenceValue:
    """Szó szerint az eredeti microgpt Value osztálya (a karpathy gist-ből)."""

    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, _ReferenceValue) else _ReferenceValue(other)
        return _ReferenceValue(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, _ReferenceValue) else _ReferenceValue(other)
        return _ReferenceValue(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return _ReferenceValue(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return _ReferenceValue(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return _ReferenceValue(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return _ReferenceValue(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


class _ReferenceMicroGPT:
    """Az eredeti (Value-alapú) microgpt hű másolata, a karpathy gist alapján."""

    OLD_TO_NEW = {
        "wte": "token_embedding",
        "wpe": "position_embedding",
        "lm_head": "lm_head",
    }
    for _li in range(microgpt.n_layer):
        OLD_TO_NEW[f"layer{_li}.attn_wq"] = f"layer{_li}.query_w"
        OLD_TO_NEW[f"layer{_li}.attn_wk"] = f"layer{_li}.key_w"
        OLD_TO_NEW[f"layer{_li}.attn_wv"] = f"layer{_li}.value_w"
        OLD_TO_NEW[f"layer{_li}.attn_wo"] = f"layer{_li}.output_w"
        OLD_TO_NEW[f"layer{_li}.mlp_fc1"] = f"layer{_li}.fc1_w"
        OLD_TO_NEW[f"layer{_li}.mlp_fc2"] = f"layer{_li}.fc2_w"

    def __init__(self, seed=0):
        _random.seed(seed)
        n_layer, n_embd = microgpt.n_layer, microgpt.n_embd
        block_size, vocab_size = microgpt.block_size, microgpt.vocab_size
        n_head, head_dim = microgpt.n_head, microgpt.head_dim
        self.n_layer, self.n_embd = n_layer, n_embd
        self.n_head, self.head_dim = n_head, head_dim
        def matrix(nout, nin, std=0.08):
            return [
                [_ReferenceValue(_random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)
            ]
        self.state_dict = {
            "wte": matrix(vocab_size, n_embd),
            "wpe": matrix(block_size, n_embd),
            "lm_head": matrix(vocab_size, n_embd),
        }
        for i in range(n_layer):
            self.state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
            self.state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
            self.state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)

    @staticmethod
    def _linear(x, w):
        return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

    @staticmethod
    def _softmax(logits):
        max_val = max(v.data for v in logits)
        exps = [(v - max_val).exp() for v in logits]
        total = sum(exps)
        return [e / total for e in exps]

    @staticmethod
    def _rmsnorm(x):
        ms = sum(xi * xi for xi in x) / len(x)
        scale = (ms + 1e-5) ** -0.5
        return [xi * scale for xi in x]

    def gpt(self, token_id, pos_id, keys, values):
        sd = self.state_dict
        x = [t + p for t, p in zip(sd["wte"][token_id], sd["wpe"][pos_id])]
        x = self._rmsnorm(x)
        for li in range(self.n_layer):
            x_residual = x
            x = self._rmsnorm(x)
            q = self._linear(x, sd[f"layer{li}.attn_wq"])
            k = self._linear(x, sd[f"layer{li}.attn_wk"])
            v = self._linear(x, sd[f"layer{li}.attn_wv"])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs : hs + self.head_dim]
                k_h = [ki[hs : hs + self.head_dim] for ki in keys[li]]
                v_h = [vi[hs : hs + self.head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / self.head_dim ** 0.5
                    for t in range(len(k_h))
                ]
                attn_weights = self._softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)
            x = self._linear(x_attn, sd[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]
            x_residual = x
            x = self._rmsnorm(x)
            x = self._linear(x, sd[f"layer{li}.mlp_fc1"])
            x = [xi.relu() for xi in x]
            x = self._linear(x, sd[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]
        return self._linear(x, sd["lm_head"])


def test_backward_matches_reference_autograd():
    """A numpy backward gradienseinek egyezése az eredeti Value-autograddal."""
    ref = _ReferenceMicroGPT(seed=0)

    # Feltöltjük az én numpy paramétereimet a referencia értékeivel
    microgpt.state_dict = {}
    for old_key, mat in ref.state_dict.items():
        new_key = _ReferenceMicroGPT.OLD_TO_NEW[old_key]
        microgpt.state_dict[new_key] = np.array(
            [[v.data for v in row] for row in mat], dtype=np.float32
        )
    microgpt.params = [microgpt.state_dict[k] for k in microgpt.param_keys]

    doc = microgpt.docs[0]
    toks = [microgpt.BOS] + [microgpt.stoi[c] for c in doc] + [microgpt.BOS]
    n = min(microgpt.block_size, len(toks) - 1)

    # 1) Referencia: eredeti pozíció-loop + Value-autograd
    keys = [[] for _ in range(microgpt.n_layer)]
    values = [[] for _ in range(microgpt.n_layer)]
    losses, ref_logits = [], []
    for pos in range(n):
        logits = ref.gpt(toks[pos], pos, keys, values)
        ref_logits.append([v.data for v in logits])
        probs = ref._softmax(logits)
        losses.append(-probs[toks[pos + 1]].log())
    loss = sum(losses) * (1 / n)
    loss.backward()

    ref_grads = {}
    for old_key, mat in ref.state_dict.items():
        new_key = _ReferenceMicroGPT.OLD_TO_NEW[old_key]
        ref_grads[new_key] = np.array([[v.grad for v in row] for row in mat])

    # 2) Az én numpy implementációm
    logits2, cache = microgpt.forward(toks[:n])
    probs2 = microgpt.softmax(logits2, axis=-1)
    idx = np.arange(n)
    grads = {k: np.zeros_like(p) for k, p in zip(microgpt.param_keys, microgpt.params)}
    dlogits = probs2.copy()
    dlogits[idx, toks[1 : n + 1]] -= 1.0
    dlogits /= n
    microgpt.backward(cache, dlogits, grads)

    # A forward (logitek) és a backward (gradiensek) is egyezzen
    np.testing.assert_allclose(logits2, np.stack(ref_logits), rtol=1e-5, atol=1e-5)
    for key in microgpt.param_keys:
        np.testing.assert_allclose(grads[key], ref_grads[key], rtol=1e-4, atol=1e-6)


def test_training_reduces_loss_and_is_deterministic():
    microgpt.num_steps = 60
    np.random.seed(42)
    microgpt.reset_params()
    l1 = microgpt.train()
    np.random.seed(42)
    microgpt.reset_params()
    l2 = microgpt.train()
    np.testing.assert_array_equal(l1, l2)
    assert l1[-1] < l1[0]


def test_generate_respects_prefix_and_length():
    microgpt.num_steps = 60
    np.random.seed(42)
    microgpt.reset_params()
    microgpt.train()
    name = microgpt.generate("em", temperature=0.5)
    assert isinstance(name, str)
    assert name.startswith("em")
    assert len(name) <= microgpt.block_size
    assert all(ch in microgpt.uchars for ch in name)


def test_generate_from_empty_stops_at_bos():
    microgpt.num_steps = 60
    np.random.seed(42)
    microgpt.reset_params()
    microgpt.train()
    for _ in range(5):
        name = microgpt.generate("", temperature=0.5)
        assert len(name) < microgpt.block_size
        assert all(ch in microgpt.uchars for ch in name)
