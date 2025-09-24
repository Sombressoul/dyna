---

Разложение на 1D Theta-суммы невозможно в общем случае из-за наличия поворота ковариации, не сохраняющего целочисленную решётку:

$$
\boxed{
\Sigma_j := \mathcal{R}(\vec{d}_j)^\dagger \cdot \mathrm{diag}(\overbrace{\sigma_j^{\parallel}, \underbrace{\sigma_j^{\perp}, \dotsc, \sigma_j^{\perp}}_{N-1}}^{\text{pos}}, \overbrace{\sigma_j^{\parallel}, \underbrace{\sigma_j^{\perp}, \dotsc, \sigma_j^{\perp}}_{N-1}}^{\text{dir}}) \cdot \mathcal{R}(\vec{d}_j)
}
$$

Вердикт: 1D-факторизация на тэта-произведение невозможна.

---

Пуассоновская ресуммация даёт строгую эквивалентность между решёточной суммой в конфигурационном прострнастве и "дуальной" суммой в импульсном.

---

Возможный путь: блочная факторизация pos ⊕ dir и пуассоновский (Ewald) сплит.

$\mathcal{R}$ **не смешивает** подпространства $\text{pos}$ и $\text{dir}$, блочная факторизация по разложению $\mathbb{C}^{2N}\cong \mathbb{C}^N_{\text{pos}}\oplus\mathbb{C}^N_{\text{dir}}$ допустима.

Имеем Меллин-связь (по сути, свёртку Дирихле двух рядов).

Тогда:

$$
\Theta_A(t,\mathbf b)=\frac{1}{t^{N}\sqrt{\det A}}\sum_{\mathbf k\in\mathbb{Z}^{2N}}
\exp\!\big(-\pi\,\mathbf k^\top A^{-1}\mathbf k/t + 2\pi i\,\mathbf k\!\cdot\!\mathbf b\big).
$$

— **строго эквивалентное** разбиение суммы на две быстро сходящиеся части; где каждая часть ещё и **блочно факторизуется**.

---

Имеем

$$
A_j=\begin{pmatrix}
A_j^{(\text{pos})} & 0\\[2pt]
0 & A_j^{(\text{dir})}
\end{pmatrix},
\qquad
A_j^{(\text{pos})}=A_j^{(\text{dir})}
=R(\vec d_j)\,\mathrm{diag}(\sigma_j^{\parallel-1},\underbrace{\sigma_j^{\perp-1},\dots,\sigma_j^{\perp-1}}_{N-1})\,R(\vec d_j)^\dagger.
$$

Многомерная тэта с характеристикой распадается в произведение двух $N$-мерных — дуальное (Пуассоновское) представление:

$$
\Theta_{A_j}(t,\mathbf b_j)
=\frac{1}{t^{N}\sqrt{\det A_j}}
\sum_{\mathbf k\in\mathbb{Z}^{2N}}
\exp\!\Big(-\pi\,\mathbf k^\top A_j^{-1}\mathbf k/t + 2\pi i\,\mathbf k\!\cdot\!\mathbf b_j\Big)
$$

Ewald-часть также мультипликативно распадается:

$$
=\Bigg[\frac{1}{t^{\frac N2}\sqrt{\det A_j^{(\text{pos})}}}
\sum_{\mathbf k\in\mathbb{Z}^{N}}
\exp\!\Big(-\pi\,\mathbf k^\top (A_j^{(\text{pos})})^{-1}\mathbf k/t + 2\pi i\,\mathbf k\!\cdot\!\mathbf b_{\text{pos}}\Big)\Bigg]
\cdot
\Bigg[\frac{1}{t^{\frac N2}\sqrt{\det A_j^{(\text{dir})}}}
\sum_{\mathbf \ell\in\mathbb{Z}^{N}}
\exp\!\Big(-\pi\,\mathbf \ell^\top (A_j^{(\text{dir})})^{-1}\mathbf \ell/t + 2\pi i\,\mathbf \ell\!\cdot\!\mathbf b_{\text{dir}}\Big)\Bigg].
$$

Связь Меллина это означает, что неоднородная Эпштейнова зета $Z_{A_j,\mathbf b_j}(s)$ для всего $2N$ эквивалентна **Меллин-свёртке** двух $N$-мерных компонент (соответствующих $\text{pos}$ и $\text{dir}$).

*ВАЖНО 01*: Мы используем две эквивалентные тэта-репрезентации и комбинируем их по схеме Ewald, т.е. с корректирующими членами и исключением нулевой моды, чтобы избежать двойного счёта.
*ВАЖНО 02*: Продукт тэта-функций порождает **Меллин-свёртку** соответствующих плотностей по $t$. Это **не** «произведение» двух зет, а именно свёртка на уровне интегрального представления. Чтобы не было смешения с «свёрткой Дирихле».

---

Рассмотрим один блок разложения ($\text{dir}$):

$$
\Bigg[\frac{1}{t^{\frac N2}\sqrt{\det A_j^{(\text{dir})}}}
\sum_{\mathbf \ell\in\mathbb{Z}^{N}}
\exp\!\Big(-\pi\,\mathbf \ell^\top (A_j^{(\text{dir})})^{-1}\mathbf \ell/t + 2\pi i\,\mathbf \ell\!\cdot\!\mathbf b_{\text{dir}}\Big)\Bigg].
$$

— он точно переписывается в вид:

$$
\boxed{
\;t^{-\,\frac N2}\,\sqrt{\sigma_j^{\parallel}\,\sigma_j^{\perp\,\,N-1}}\;
\sum_{\ell\in\mathbb{Z}^N}
\exp\!\Big(
-\tfrac{\pi}{t}\big[\sigma_j^{\perp}\,\|\ell\|_2^2+(\sigma_j^{\parallel}-\sigma_j^{\perp})\,|\langle \ell, r_1\rangle|^2\big]
\;+\;2\pi i\,\ell\!\cdot\!\mathbf b_{\mathrm{dir}}
\Big)\;.
}
$$

Тогда итоговый вид для эффективного счёта:

$$
t^{-\tfrac N2}\sqrt{\sigma_\parallel\,\sigma_\perp^{\,N-1}}
\sum_{\ell\in\mathbb{Z}^N}
\exp\!\Big(-\tfrac{\pi}{t}\,[\sigma_\perp\,s + (\sigma_\parallel-\sigma_\perp)\,|p|^2]\Big)\;\phi(\ell),
$$


где $s=\sum \ell_k^2$, $p=\sum \ell_k r_{1,k}$, $\phi(\ell)=\prod u_k^{\ell_k}$,
а $s,p,\phi$ **обновляются инкрементально** при пробеге по $\ell$.

При этом $L_\infty$-отсечка и оценка хвоста:

$$
\sum_{\|\ell\|_\infty>L} 
\exp\!\Big(-\tfrac{\pi}{t}\,\ell^\top (A_j^{(\mathrm{dir})})^{-1}\ell\Big)
\;\le\;
\sum_{m=L+1}^\infty
A_N(m)\,\exp\!\Big(-\tfrac{\pi}{t}\,\sigma_{\min}\,m^2\Big),
$$

где $A_N(m)=(2m+1)^N-(2m-1)^N$, $\sigma_{\min}=\min(\sigma_\parallel,\sigma_\perp)$.

---

**ВАЖНОЕ УТОЧНЕНИЕ**

В CPSF решётка действует **только на позиционный блок**. Значит, при поассоновской ресуммации сумма появляется только по позиционным модам; «дирекционный» блок даёт лишь мультипликативный гауссов фактор (без собственной решёточной суммы). 

Так как периодизация действует по позиции, на дуальной стороне возникает **одна** $N$-комплексная (или $2N$-реальная) тэта-сумма по позиционным модам **с фазой**, зависящей от дробной части позиционного сдвига (в нашей нотации — от $\tilde z - \tilde z_j$ после поворота). Дирекционный вклад $\delta\vec d$ входит как гауссов множитель в экспоненте (через рангово-1 поправку внутри $S_0$), что полностью согласуется с формулой для `q(w)` в коде.

Пуассоновская ресуммация/Евальдов сплит применяются к **той части**, где у нас действительно есть решёточная периодизация — то есть к позиционному $2N$-реальному (или $N$-комплексному) блоку. Это строго совместимо с каноном: `R_ext` блочно-диагональна; `Sigma` — тоже; `q` и `rho` определены для $w=[u;v]$ при фиксированном $v=\delta\vec d$.

Ошибочную трактовку «две независимые $N$-тэта-суммы (pos⊕dir)» снимаем: в текущем каноне решётка не трогает `dir`-часть.

---
