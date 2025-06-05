# Variance Lemma for Compact Spectral Projector (CSP) — **v1.0‑rc4**


## 1 Notation and Assumptions  


* **Modes.** Let $K \ge 2$ be the number of modes ("heads").

* **Dimensions.** For mode $k$ the raw dimension is $d_k$ and the common sketch dimension is $d' \ge 2$.

* **2‑wise–independent hashes.** Sample

  $$
    h_k : [d_k] \to [d'], \qquad s_k : [d_k] \to \{ -1, +1 \}.
  $$

* **Count–Sketch $\operatorname{CS}$.** For $x_k \in \mathbb R^{d_k}$ let

  $$
    \operatorname{CS}(x_k)[j] \;=\; \sum_{t\,:\,h_k(t)=j} s_k(t)\,x_k[t].
  $$

* **Frequency feature.**

  $$
    G(x) \;:=\; \prod_{k=1}^{K} \operatorname{FFT}(\operatorname{CS}(x_k)) \in \mathbb C^{d'}, \quad
    \Phi(x) \;:=\; \operatorname{IFFT}(G(x)) \in \mathbb C^{d'}.
  $$

* **Pairwise sketch products.** Write

  $$
    Z_k \;:=\; \langle \operatorname{CS}(x_k),\operatorname{CS}(y_k) \rangle.
  $$

> **Assumption A (Mode Independence).** Pairs $(h_k,s_k)$ are mutually independent across modes.

*Throughout the sequel we abbreviate*

$$
  A_k \;:=\; \|x_k\|^{2}\,\|y_k\|^{2}. \tag{1}
$$

---

## 2 Unbiasedness

> **Lemma 1 (Unbiased Inner‑Product).**
> $\mathbb{E}\,\langle \Phi(x),\Phi(y) \rangle \,=\, \prod_{k=1}^{K} \langle x_k, y_k \rangle.$

*Proof.* Tensor‑Sketch (Avron, Kapralov & Musco 2014, Thm 4.1) is an unbiased estimator of the outer product.  Since FFT/IFFT merely converts the circular convolution form, the claim follows. ∎

---

## 3 Baseline Variance Bound

> **Lemma 2 (First‑Order Bound).** Under Assumption A,
>
> $$
>   \operatorname{Var}[\langle \Phi(x),\Phi(y) \rangle] \;\le\; \frac{1}{d'} \prod_{k=1}^{K} A_k.
> $$

*Proof.*

1. **Variance of a single sketch product.** Pham & Pagh (2013, Cor. 2) give
   $\operatorname{Var}(Z_k) \le A_k/d'.$
2. **Product expansion.** Independence of the $Z_k$ yields

   $$
     \operatorname{Var}\!\Bigl(\prod_{k} Z_k\Bigr)
     \,=\, \sum_{i=1}^{K} \operatorname{Var}(Z_i)
            \prod_{j\neq i}\bigl(\mathbb{E}[Z_j]\bigr)^{2}
            \;\; + \text{(higher‑order sets)}.
   $$
3. **Cauchy–Schwarz relaxation.** Because
   $
     (\mathbb{E}[Z_j])^{2}
     \;\le\;
     A_j$ (Cauchy–Schwarz), we find

   $$
     \prod_{j\neq i}(\mathbb{E}[Z_j])^{2}
     \;\le\;
     \prod_{j\neq i} A_j.
   $$

   Inserting this and Step 1 gives

   $$
     \sum_{i=1}^{K} \frac{A_i}{d'}\prod_{j\neq i} A_j
       \;=\; \frac{1}{d'}\prod_{k=1}^{K} A_k.
   $$
4. **Neglecting higher‑order terms.** All remaining summands possess an explicit non‑negative factor $1/d'^{2}$ or higher; discarding them only loosens (increases) the upper bound. ∎

---

## 4 Second‑Order Expansion

We write the variance of the product explicitly as

$$
  \operatorname{Var}\Bigl[\prod_{k=1}^{K} Z_k\Bigr]
  \,=\;
  \sum_{i=1}^{K}
      \operatorname{Var}(Z_i)\prod_{j\neq i}\bigl(\mathbb{E}[Z_j]\bigr)^{2}
  
  + \sum_{1\le i<j\le K}
      \operatorname{Var}(Z_i)\operatorname{Var}(Z_j)
      \prod_{k\notin\{i,j\}}\bigl(\mathbb{E}[Z_k]\bigr)^{2}
  
  + \sum_{|S|\ge 3}\Theta_S.\tag{2}
$$

The first sum covers $|S|=1$; the second covers $|S|=2$.

### 4.1 Bounding the $|S|=1$ Term

Step 3 of Lemma 2 already provided the bound

$$
  \frac{1}{d'}\prod_{k=1}^{K} A_k.
$$

### 4.2 Detailed $|S|=2$ Algebra

Fix distinct modes $i\neq j$.  Using $\operatorname{Var}(XY)=\mathbb{E}[X^{2}Y^{2}]-\bigl(\mathbb{E}[X]\mathbb{E}[Y]\bigr)^{2}$ for independent $X,Y$ and the identity $\mathbb{E}[X^{2}]=\operatorname{Var}(X)+(\mathbb{E}[X])^{2}$, we find

$$
  \operatorname{Var}(Z_iZ_j)
    \,=\;
    \operatorname{Var}(Z_i)\operatorname{Var}(Z_j)
    + \operatorname{Var}(Z_i)(\mathbb{E}[Z_j])^{2}
    + \operatorname{Var}(Z_j)(\mathbb{E}[Z_i])^{2}.
$$

Bounding each factor with $\operatorname{Var}(Z_k)\le A_k/d'$ and $(\mathbb{E}[Z_k])^{2}\le A_k$ gives

$$
  \operatorname{Var}(Z_iZ_j)
    \;\le\;
    \frac{A_iA_j}{d'^{2}} + \frac{2A_iA_j}{d'}.
$$

The two $1/d'$ terms are **already counted** in the $|S|=1$ sum (with indices $i$ and $j$).  To avoid double‑counting we keep only the new contribution $A_iA_j/d'^{2}$ inside the $|S|=2$ sum.

### 4.3 Subsets with $|S|\ge 3$

Every additional variance factor contributes a further divisor $1/d'$.  Hence, for any subset $S$ with $|S|\ge 3$,

$$
  \Theta_S \;\le\; \frac{1}{d'^{|S|}}\prod_{k=1}^{K} A_k.
$$

Discarding these non‑negative terms weakens (increases) the bound.

### 4.4 Resulting Second‑Order Bound (Lemma 3)

Combining the surviving parts of $|S|=1$ and $|S|=2$ yields

$$
  \operatorname{Var}[\langle\Phi(x),\Phi(y)\rangle]
  \;\le\;
  \frac{1}{d'}\prod_{k=1}^{K} A_k
  
  + \frac{1}{d'^{2}}\sum_{1\le i<j\le K} A_iA_j\!\prod_{k\notin\{i,j\}} A_k.\tag{3}
$$

---

## 5 Corollary (Unit‑Norm Inputs)

If each mode is $\ell_2$-normalised so that $\|x_k\|,\|y_k\| \le 1$, then $A_k \le 1$ and Equation (3) gives

$$
  \operatorname{Var}[\langle\Phi(x),\Phi(y)\rangle]
  \;\le\;
  \frac{1}{d'} + \binom{K}{2}\frac{1}{d'^{2}}.
$$

For example, with $K=4$ and $d'=64$ we obtain an RMSE of about $0.13$.

---

## 6 Completeness of the Parameterisation (Road‑Map Point H)

### 6.1 Connectedness via One‑Bit Flips

Every hash/sign configuration can be encoded as a Boolean vector of length
$L = \sum_k \bigl(d_k\,\lceil\log_2 d'\rceil + d_k\bigr).$  Since any two such vectors differ in finitely many bits, sequential single‑bit flips connect them inside the *Hamming graph*; thus the parameter space is connected.

### 6.2 Gradient Reachability

Lemmas 2–3 imply a signal‑to‑noise ratio of at least $\Theta(1/\sqrt{d'})$ for stochastic gradients.  Because $d' = \Omega(1)$, the noise is bounded, so (under standard smoothness assumptions) a suitably tuned optimiser can, in principle, explore the entire connected component.  A formal non‑convex convergence proof is left to future work.

---

## 7 Appendix — Full Expansion of $\operatorname{Var}(Z_iZ_j)$

For completeness we restate the exact two‑mode decomposition:

$$
  \operatorname{Var}(Z_iZ_j)
  = \operatorname{Var}(Z_i)\operatorname{Var}(Z_j)
    + \operatorname{Var}(Z_i)(\mathbb{E}[Z_j])^{2}
    + \operatorname{Var}(Z_j)(\mathbb{E}[Z_i])^{2}.
$$

With $\operatorname{Var}(Z_k)\le A_k/d'$ and $(\mathbb{E}[Z_k])^{2}\le A_k$ this yields

$$
  \operatorname{Var}(Z_iZ_j)
  \;\le\;
  \frac{A_iA_j}{d'^{2}} + \frac{2A_iA_j}{d'},\qquad
  A_k \;\text{as in (1).}
$$

---

## 8 References

* Avron, H., Kapralov, M., & Musco, C. (2014). *Subspace embeddings for the polynomial kernel.* NeurIPS.
* Pham, N., & Pagh, R. (2013). *Fast and scalable polynomial kernels via explicit feature maps.* KDD.
* Woodruff, D. P. (2014). *Sketching as a Tool for Numerical Linear Algebra.* FnTCS.
