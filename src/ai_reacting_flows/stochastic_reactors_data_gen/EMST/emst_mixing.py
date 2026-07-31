"""
EMST micro-mixing (Euclidean Minimum Spanning Tree) -- portage Python fidele
et VECTORISE du modele intermittent de Subramaniam & Pope,
Combust. Flame 115:487-514 (1998), implementation Fortran Z. Ren & S. B. Pope
(Cornell, 2002).

Version optimisee pour usage HPC (milliers de particules par ensemble) :
  - topologie d'arbre precalculee (ordre topologique, parent, arete-parent) :
    ne depend ni de eta ni de la dimension, calculee une seule fois par pas ;
  - `_implct` vectorise sur toutes les dimensions simultanement, sans boucle
    Python sur les noeuds a l'interieur d'une dimension ;
  - MST euclidien exact par Prim vectorise (mise a jour NumPy des lowcost),
    sans materialiser la matrice de distances complete pour les grands n ;
  - poids de sous-arbre (edgewt) cumules en une passe topologique.

Reproduit fidelement : variable d'age intermittente (state), coefficients
d'arete ponderes B(e)=2*w_sub, integration implicite sur l'arbre, calage de eta
par recherche de racine vers exp(-omdt), repli IEM, evolution de state.

Convention msubn = 1 (un seul sous-ensemble), comme dans emst_scl.

Interface principale (drop-in Fortran) :
    emst(mode, f, state, wt, omdt, fscale, cvars, np=..., nc=...)
modifie f et state in place, retourne info (0 = succes).
"""

import numpy as np


_Z0L = 0.1666
_Z0U = 0.1667
_Z1L = 0.0178
_Z1U = 0.3157

_SDEV_SMALL = 1.0e-4
_OMDTMIN = 1.0e-6
_FPHTHR = 0.3
_RPHTHR = 1.0e-5
_GTMAX = 2.5
_IMOD = 2

_MXSRCH = 20
_MBRACK = 2

# au-dela de ce n, on n'utilise pas la matrice de distances complete (n^2).
# n=8000 -> 8000^2*8 = 512 Mo ; ajustable selon la memoire disponible.
_DENSE_MST_MAX = 8000


class EMSTMixer:
    """Melangeur EMST intermittent vectorise."""

    def __init__(self, rng=None, gtmax=_GTMAX, sdev_small=_SDEV_SMALL,
                 omdtmin=_OMDTMIN, fphthr=_FPHTHR, rphthr=_RPHTHR):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.gtmax = gtmax
        self.sdev_small = sdev_small
        self.omdtmin = omdtmin
        self.fphthr = fphthr
        self.rphthr = rphthr
        self._p0 = (_Z1L + _Z1U) / ((_Z0L + _Z0U) + (_Z1L + _Z1U))

    def init_state(self, npart):
        state = np.empty(npart, dtype=np.float64)
        u = self.rng.random(npart)
        in_tree = u <= self._p0
        nt = int(in_tree.sum())
        state[in_tree] = self._raninit(nt, _Z1L, _Z1U)
        state[~in_tree] = -self._raninit(npart - nt, _Z0L, _Z0U)
        return state

    def mix(self, phi, wt, omdt, fscale=None, state=None):
        npart, nc = phi.shape
        if state is None:
            state = self.init_state(npart)
        if omdt <= 0.0 or npart <= 1:
            return state
        if fscale is None:
            fscale = np.ones(nc, dtype=np.float64)

        g = phi / fscale

        if self._range(g) * np.exp(-omdt) < self.sdev_small / 10.0:
            self._iem(g, wt, omdt)
            self._statinc(state, omdt)
            phi[:] = g * fscale
            return state

        omtleft = omdt
        while omtleft > 0.0:
            if self._range(g) < 4.0 * self.sdev_small:
                self._iem(g, wt, omtleft)
                self._statinc(state, omtleft)
                break

            if omtleft <= self.omdtmin:
                gtemp = g.copy()
                omdtm = self._mixemst(gtemp, wt, state, 2.0 * self.omdtmin)
                if omdtm >= omtleft:
                    if self.rng.random() <= omtleft / omdtm:
                        g[:] = gtemp
                    self._statinc(state, omtleft)
                    break
                else:
                    g[:] = gtemp
                    self._statinc(state, omdtm)
                    omtleft -= omdtm
            else:
                omdtm = self._mixemst(g, wt, state, omtleft)
                self._statinc(state, omdtm)
                omtleft -= omdtm

        phi[:] = g * fscale
        return state

    def _mixemst(self, g, wt, state, dt):
        npart, nc = g.shape
        wnm = wt / wt.sum()

        mean = wnm @ g
        var = wnm @ (g - mean) ** 2
        sd = np.sqrt(np.maximum(var, 0.0))
        mxsdev = sd.max()
        dphthr = self.fphthr * np.maximum(sd, self.rphthr * mxsdev)

        avg = mean.copy()
        phi = g - avg

        idx_tree = np.where(state > 0.0)[0]
        idx_out = np.where(state <= 0.0)[0]
        idx_tree, idx_out = self._pcheck(phi, wnm, state, idx_tree, idx_out)

        nt = idx_tree.size
        if nt < 2:
            return dt

        xt = phi[idx_tree]
        wt_tree = wnm[idx_tree]

        edges = self._emst(xt)
        topo = self._build_topology(nt, edges)
        wsub = self._subtree_weights(nt, edges, wt_tree, topo)
        p_tree = wt_tree.sum()
        b = 2.0 * (wsub / p_tree)

        alpest = self._rtest(xt, wt_tree, edges, b)
        phi_tree, dtm = self._rtfini(xt, wt_tree, edges, b, topo, dt, alpest, dphthr)

        phi[idx_tree] = phi_tree
        g[:] = phi + avg
        return dtm

    # ---- MST euclidien exact (Prim) ----
    def _emst(self, x):
        n = x.shape[0]
        if n <= _DENSE_MST_MAX:
            return self._prim_dense(x)
        return self._prim_incremental(x)

    @staticmethod
    def _prim_dense(x):
        n = x.shape[0]
        sq = np.sum(x * x, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
        np.maximum(d2, 0.0, out=d2)
        # dist : distance minimale de chaque noeud hors-arbre a l'arbre ;
        # les noeuds deja dans l'arbre sont mis a +inf (pas de np.where recree).
        in_tree = np.zeros(n, dtype=bool)
        in_tree[0] = True
        dist = d2[0].copy()
        dist[0] = np.inf
        near = np.zeros(n, dtype=np.int64)
        edges = np.empty((n - 1, 2), dtype=np.int64)
        for e in range(n - 1):
            k = int(np.argmin(dist))
            edges[e, 0] = near[k]
            edges[e, 1] = k
            in_tree[k] = True
            dist[k] = np.inf
            dk = d2[k]
            # noeuds hors-arbre strictement plus proches de k que de l'arbre
            upd = (~in_tree) & (dk < dist)
            np.copyto(dist, dk, where=upd)
            near[upd] = k
        return edges

    @staticmethod
    def _prim_incremental(x):
        n = x.shape[0]
        in_tree = np.zeros(n, dtype=bool)
        in_tree[0] = True
        dist = np.sum((x - x[0]) ** 2, axis=1)
        dist[0] = np.inf
        near = np.zeros(n, dtype=np.int64)
        edges = np.empty((n - 1, 2), dtype=np.int64)
        for e in range(n - 1):
            k = int(np.argmin(dist))
            edges[e, 0] = near[k]
            edges[e, 1] = k
            in_tree[k] = True
            dist[k] = np.inf
            dk = np.sum((x - x[k]) ** 2, axis=1)
            upd = (~in_tree) & (dk < dist)
            np.copyto(dist, dk, where=upd)
            near[upd] = k
        return edges

    # ---- topologie enracinee (BFS) ----
    @staticmethod
    def _build_topology(n, edges):
        deg = np.zeros(n, dtype=np.int64)
        ii = edges[:, 0]
        jj = edges[:, 1]
        np.add.at(deg, ii, 1)
        np.add.at(deg, jj, 1)
        off = np.zeros(n + 1, dtype=np.int64)
        off[1:] = np.cumsum(deg)
        adj_node = np.empty(2 * (n - 1), dtype=np.int64)
        adj_edge = np.empty(2 * (n - 1), dtype=np.int64)
        cur = off[:-1].copy()
        for e in range(n - 1):
            i, j = int(ii[e]), int(jj[e])
            adj_node[cur[i]] = j
            adj_edge[cur[i]] = e
            cur[i] += 1
            adj_node[cur[j]] = i
            adj_edge[cur[j]] = e
            cur[j] += 1

        root = int(edges[0, 0])
        parent = np.full(n, -1, dtype=np.int64)
        par_edge = np.full(n, -1, dtype=np.int64)
        order = np.empty(n, dtype=np.int64)
        visited = np.zeros(n, dtype=bool)
        order[0] = root
        visited[root] = True
        head, tail = 0, 1
        while head < tail:
            u = int(order[head])
            head += 1
            for p in range(off[u], off[u + 1]):
                v = int(adj_node[p])
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    par_edge[v] = int(adj_edge[p])
                    order[tail] = v
                    tail += 1
        return {"order": order, "parent": parent, "par_edge": par_edge, "root": root}

    @staticmethod
    def _subtree_weights(n, edges, w, topo):
        order = topo["order"]
        parent = topo["parent"]
        par_edge = topo["par_edge"]
        node_w = w.copy()
        subtree = np.zeros(n - 1, dtype=np.float64)
        for k in range(n - 1, -1, -1):
            u = int(order[k])
            pe = par_edge[u]
            if pe != -1:
                subtree[pe] += node_w[u]
                node_w[parent[u]] += node_w[u]
        total = w.sum()
        return np.minimum(subtree, total - subtree)

    # ---- integration implicite VECTORISEE ----
    @staticmethod
    def _implct(phi, w, edges, b, eta, topo):
        n, nc = phi.shape
        order = topo["order"]
        parent = topo["parent"]
        par_edge = topo["par_edge"]

        S = np.zeros(n, dtype=np.float64)
        np.add.at(S, edges[:, 0], b)
        np.add.at(S, edges[:, 1], b)
        diag0 = w + eta * S

        rhs = w[:, None] * phi

        d_coef = np.zeros((n, nc), dtype=np.float64)
        e_coef = np.zeros(n, dtype=np.float64)
        acc_diag = diag0.copy()
        acc_rhs = rhs.copy()

        for k in range(n - 1, 0, -1):
            u = int(order[k])
            pe = int(par_edge[u])
            bpe = b[pe]
            inv = 1.0 / acc_diag[u]
            d_coef[u] = inv * acc_rhs[u]
            e_coef[u] = inv * eta * bpe
            p = int(parent[u])
            acc_rhs[p] += eta * bpe * d_coef[u]
            acc_diag[p] -= eta * bpe * e_coef[u]

        root = int(topo["root"])
        d_coef[root] = acc_rhs[root] / acc_diag[root]
        e_coef[root] = 0.0

        sol = np.empty((n, nc), dtype=np.float64)
        sol[root] = d_coef[root]
        for k in range(1, n):
            u = int(order[k])
            p = int(parent[u])
            sol[u] = d_coef[u] + e_coef[u] * sol[p]
        return sol

    @staticmethod
    def _rtest(phi, w, edges, b):
        m = w @ phi
        varfn = float(np.sum(w @ (phi - m) ** 2))
        if varfn <= 0.0:
            return 1.0
        di = phi[edges[:, 0]] - phi[edges[:, 1]]
        dsum = float(np.sum(b * np.sum(di * di, axis=1)))
        return max(dsum / (2.0 * varfn), 1.0e-30)

    def _rtfini(self, phi, w, edges, b, topo, dt, alpest, dphthr):
        def varfunc(p):
            m = w @ p
            return float(np.sum(w @ (p - m) ** 2))

        varfn = varfunc(phi)
        varfnd = varfn * np.exp(-dt)

        eta2 = 1.08 * alpest * dt
        ainc = 1.05
        phitmp = phi
        fl = None
        etal = eta2

        for _ in range(_MBRACK):
            etal = eta2
            fh = varfn - varfnd
            isrch = 0
            bracketed = False
            while True:
                phitmp = self._implct(phi, w, edges, b, etal, topo)
                fl = varfunc(phitmp) - varfnd
                if fl > 0.0:
                    if isrch < _MXSRCH:
                        if fl > 0.4 * fh:
                            etal = 1.10 * etal
                        else:
                            etal = ainc * etal / (1.0 - fl / fh)
                        isrch += 1
                        continue
                    else:
                        break
                else:
                    bracketed = True
                    break
            if bracketed:
                break
            eta2 = 0.99 * eta2
            ainc *= 1.05

        while True:
            dphimx = np.max(np.abs(phitmp - phi), axis=0)
            if np.all(dphimx <= dphthr):
                dtmtemp = np.log(varfn / (fl + varfnd))
                if dtmtemp > dt:
                    if self.rng.random() < dt / dtmtemp:
                        phi = phitmp
                    return phi, dt
                else:
                    return phitmp, dtmtemp
            etal = etal / 2.0
            phitmp = self._implct(phi, w, edges, b, etal, topo)
            fl = varfunc(phitmp) - varfnd

    def _pcheck(self, phi, w, state, idx_tree, idx_out):
        m_all = w @ phi
        varfn = float(np.sum(w @ (phi - m_all) ** 2))
        idx_tree = list(idx_tree)
        idx_out = list(idx_out)

        def tree_stats():
            it = np.array(idx_tree, dtype=np.int64)
            p = w[it].sum()
            if p <= 0.0:
                return p, 0.0
            wl = w[it] / p
            m = wl @ phi[it]
            vt = float(np.sum(wl @ (phi[it] - m) ** 2))
            return p, vt

        p, varfnT = tree_stats()
        pmin = min(varfn / (varfnT * self.gtmax), 1.0) if varfnT > 0.0 else 1.0
        if p >= pmin or not idx_out:
            return (np.array(idx_tree, dtype=np.int64),
                    np.array(idx_out, dtype=np.int64))

        idx_out.sort(key=lambda k: abs(state[k]))
        while p < pmin and idx_out:
            k = idx_out.pop(0)
            state[k] = _Z1L + (_Z1U - _Z1L) * self.rng.random()
            idx_tree.append(k)
            p, varfnT = tree_stats()
            pmin = min(varfn / (varfnT * self.gtmax), 1.0) if varfnT > 0.0 else 1.0

        return (np.array(idx_tree, dtype=np.int64),
                np.array(idx_out, dtype=np.int64))

    @staticmethod
    def _iem(g, wt, omdt):
        wtsum = wt.sum()
        if wtsum <= 0.0:
            return
        decay = 1.0 - np.exp(-0.5 * omdt)
        gmean = (wt @ g) / wtsum
        g -= decay * (g - gmean)

    def _statinc(self, state, ds):
        u = self.rng.random(state.size)
        flip = np.abs(state) <= ds
        neg = flip & (state < 0.0)
        pos = flip & (state >= 0.0)
        state[neg] = _Z1L + (_Z1U - _Z1L) * u[neg]
        state[pos] = -(_Z0L + (_Z0U - _Z0L) * u[pos])
        keep_neg = (~flip) & (state < 0.0)
        keep_pos = (~flip) & (state >= 0.0)
        state[keep_neg] += ds
        state[keep_pos] -= ds

    def _raninit(self, n, a, b):
        if n <= 0:
            return np.empty(0, dtype=np.float64)
        return a + (b - a) * self.rng.random(n)

    @staticmethod
    def _range(g):
        return float(np.max(g.max(axis=0) - g.min(axis=0)))


# ==== Interface style Fortran (drop-in) ====

_MODULE_RNG = np.random.default_rng()


def emst_seed(seed):
    global _MODULE_RNG
    _MODULE_RNG = np.random.default_rng(seed)


def emst(mode, f, state, wt, omdt, fscale, cvars, nc=None, **kwargs):
    """Reproduit emst(mode,np,nc,f,state,wt,omdt,fscale,cvars,info).
    Modifie f et state in place, retourne info (0=succes, <0=erreur)."""
    np_count = kwargs.pop("np", None)

    if mode not in (1, 2):
        return -1
    f = np.asarray(f)
    npart = f.shape[0] if np_count is None else int(np_count)
    ncompo = f.shape[1] if nc is None else int(nc)
    if npart < 0:
        return -2
    if npart == 0:
        return 0
    if npart == 1 and mode == 2:
        return 0

    cvars = np.asarray(cvars, dtype=np.float64).ravel()
    icheck = 1 if (cvars.size >= 1 and round(float(cvars[0])) == 1) else 0
    sdev_small = float(cvars[2]) if (cvars.size >= 3 and cvars[2] > 0.0) else _SDEV_SMALL
    c4 = float(cvars[3]) if cvars.size >= 4 else 0.0
    omdtmin = c4 if (1.0e-7 < c4 <= 1.0e-2) else _OMDTMIN
    fphthr = float(cvars[4]) if (cvars.size >= 5 and cvars[4] > 0.0) else _FPHTHR
    rphthr = float(cvars[5]) if (cvars.size >= 6 and cvars[5] > 0.0) else _RPHTHR

    mixer = EMSTMixer(rng=_MODULE_RNG, sdev_small=sdev_small,
                      omdtmin=omdtmin, fphthr=fphthr, rphthr=rphthr)

    if mode == 1:
        state[:] = mixer.init_state(npart).astype(state.dtype, copy=False)
        return 0

    if omdt == 0.0:
        return 0
    if omdt < 0.0:
        return -7
    if ncompo < 1:
        return -3

    wt_arr = np.asarray(wt, dtype=np.float64).ravel()
    fscale_arr = np.asarray(fscale, dtype=np.float64).ravel()
    state_arr = np.asarray(state, dtype=np.float64).ravel().copy()

    if icheck == 1:
        if wt_arr.min() < 0.0:
            return -4
        if state_arr.max() > _Z1U or state_arr.min() < -_Z0U:
            return -5
        if fscale_arr.min() <= 0.0:
            return -6

    f64 = f.astype(np.float64, copy=True)
    mixer.mix(f64, wt_arr, float(omdt), fscale=fscale_arr, state=state_arr)

    f[:] = f64.astype(f.dtype, copy=False)
    state[:] = state_arr.astype(state.dtype, copy=False)
    return 0
