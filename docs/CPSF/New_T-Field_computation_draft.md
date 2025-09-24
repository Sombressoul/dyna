# 0) Вход, допущения, константы

* Размерность $N\ge 2$.
* Входы (для каждого $j$): $z\in\mathbb C^N,\ z_j\in\mathbb C^N,\ \vec d\in\mathbb C^N,\ \vec d_j\in\mathbb C^N,\ \alpha_j\in\mathbb R_{>0},\ \sigma_{\parallel j},\sigma_{\perp j}\in\mathbb R_{>0},\ \hat T_j\in\mathbb C^S$.
* Конструкции канона (как в CPSF.md, `core_math.py`):

  * $\delta\vec d:=\mathrm{delta\_vec\_d}(\vec d,\vec d_j)$.
  * $\mathrm{iota}(u,v)=[u;v]\in\mathbb C^{2N}$.
  * $R(\vec d_j)\in \mathrm{U}(N)$, $\mathcal R(\vec d_j):=\mathrm{diag}(R(\vec d_j),R(\vec d_j))$.
  * Диагональ $D_j=\mathrm{diag}(\sigma_{\parallel j},\sigma_{\perp j},\dots;\ \sigma_{\parallel j},\sigma_{\perp j},\dots)$.
  * Ковариация $\Sigma_j = \mathcal R^\dagger D_j\,\mathcal R$ (блочно-диагональна).
  * Квадратичная форма $q_j(w)=\langle \Sigma_j^{-1}w,w\rangle$, ядро $\rho_j(q)=\exp(-\pi q)$.
* **Периодизация действует только по позиции.** Решётка $\Lambda=\mathbb Z^{2N}$ сдвигает $\delta z := z - z_j$ по $(\Re,\Im)$ координатам позиции; $\delta\vec d$ не периодизуется.

---

# 1) Точное определение $\eta_j$ и факторизация pos ⊕ dir

Каноническая сумма:

$$
\boxed{
\eta_j(z,\vec d)\;=\;
\sum_{n\in\mathbb Z^{2N}}
\rho_j\!\Big(q_j\big(\mathrm{iota}(\delta z+n,\ \delta\vec d)\big)\Big).
}
$$

Из блочно-диагональной структуры $\Sigma_j^{-1}=\mathrm{diag}(A^{(\mathrm{pos})}_j,\ A^{(\mathrm{dir})}_j)$ следует

$$
q_j\big(\mathrm{iota}(\delta z+n,\ \delta\vec d)\big)
=\,(\delta z+n)^\dagger A^{(\mathrm{pos})}_j(\delta z+n)\;+\;\delta\vec d^\dagger A^{(\mathrm{dir})}_j\,\delta\vec d.
$$

Так как $\delta\vec d$ фиксированно по суммированию, имеем точную факторизацию:

$$
\boxed{
\eta_j
=\underbrace{\exp\!\big(-\pi\,\delta\vec d^\dagger A^{(\mathrm{dir})}_j\,\delta\vec d\big)}_{\displaystyle C^{(\mathrm{dir})}_j\ \text{— гауссов множитель}}
\cdot
\underbrace{\sum_{n\in\mathbb Z^{2N}} \exp\!\big(-\pi\,(\delta z+n)^\dagger A^{(\mathrm{pos})}_j(\delta z+n)\big)}_{\displaystyle \Theta^{(\mathrm{pos})}_j\ (\text{позиционная решёточная сумма})}.
}
$$

Здесь

$$
A^{(\mathrm{pos})}_j=A^{(\mathrm{dir})}_j
=\ R(\vec d_j)\,\mathrm{diag}(\sigma_{\parallel j}^{-1},\sigma_{\perp j}^{-1},\ldots)\,R(\vec d_j)^\dagger.
$$

(Ранг-1 анизотропия вдоль первого столбца $r_1$ матрицы $R(\vec d_j)$ сохраняется внутри каждого блока.)

---

# 2) Дуальная (Пуассоновская) репрезентация позиционной суммы

Обозначим $d=2N$ и

$$
b_{\mathrm{pos}}:=\delta z \ \ (\text{в координатах торовой фундаментальной области; т.е. по модулю }\mathbb Z^{2N}).
$$

Тогда для любого $t>0$ справедлива точная тэта-формула:

$$
\boxed{
\Theta^{(\mathrm{pos})}_j
=\frac{1}{t^{d/2}\sqrt{\det A^{(\mathrm{pos})}_j}}\;
\sum_{k\in\mathbb Z^{2N}}
\exp\!\Big(-\frac{\pi}{t}\,k^\top (A^{(\mathrm{pos})}_j)^{-1}k\Big)\;
\exp\!\big(2\pi i\,k\!\cdot\! b_{\mathrm{pos}}\big).
}
$$

Здесь $(A^{(\mathrm{pos})}_j)^{-1}=R(\vec d_j)\,\mathrm{diag}(\sigma_{\parallel j},\sigma_{\perp j},\ldots)\,R(\vec d_j)^\dagger$.

> **Примечание (о “Ewald”).** Реальная и дуальная формы — **эквивалентные** представления одной и той же суммы. Их *можно* комбинировать по схеме Ewald (с исключением нулевой моды и корректирующим членом), но это **не обязательно**: для вычисления достаточно выбрать одно представление и усечь его с контролем хвоста.

---

# 3) Две эквивалентные вычислительные формы $\Theta^{(\mathrm{pos})}_j$

## 3.1 Реальная форма (позиционное пространство)

$$
\boxed{
\Theta^{(\mathrm{pos})}_j
=\sum_{n\in\mathbb Z^{2N}} \exp\!\big(-\pi\,t\;(\delta z+n)^\dagger A^{(\mathrm{pos})}_j(\delta z+n)\big)
\quad\text{(тот же ряд при масштабе }t>0\text{).}
}
$$

**Дискретная оценка хвоста** при L∞-окне $\|n\|_\infty\le L$: с $\sigma_{\max j}:=\max(\sigma_{\parallel j},\sigma_{\perp j})$

$$
\sum_{\|n\|_\infty>L}\exp\!\big(-\pi\,t\,(\delta z+n)^\dagger A^{(\mathrm{pos})}_j(\delta z+n)\big)
\ \le\
\sum_{m=L+1}^\infty A_{d}(m)\,\exp\!\Big(-\pi\,t\,\frac{m^2}{\sigma_{\max j}}\Big),
$$

где $A_{d}(m)=(2m{+}1)^{d}-(2m{-}1)^{d}$.

## 3.2 Дуальная форма (импульсное пространство)

$$
\boxed{
\Theta^{(\mathrm{pos})}_j
=\frac{1}{t^{d/2}\sqrt{\det A^{(\mathrm{pos})}_j}}\;
\sum_{k\in\mathbb Z^{2N}}
\exp\!\Big(-\frac{\pi}{t}\,k^\top (A^{(\mathrm{pos})}_j)^{-1}k\Big)\;
e^{\,2\pi i\,k\cdot b_{\mathrm{pos}}}.
}
$$

**Дискретная оценка хвоста** при L∞-окне $\|k\|_\infty\le L$: с $\sigma_{\min j}:=\min(\sigma_{\parallel j},\sigma_{\perp j})$

$$
\sum_{\|k\|_\infty>L}\exp\!\Big(-\frac{\pi}{t}\,k^\top (A^{(\mathrm{pos})}_j)^{-1}k\Big)
\ \le\
\sum_{m=L+1}^\infty A_{d}(m)\,\exp\!\Big(-\frac{\pi}{t}\,\sigma_{\min j}\,m^2\Big).
$$

> **Выбор $t$.** $t>0$ выбирается так, чтобы требуемый допуск $\varepsilon$ достигался с минимальной суммарной трудоёмкостью: балансируют две экспоненты хвоста (реальная/дуальная). Допускается чисто «реальный» или чисто «дуальный» режим, а также гибридный (Ewald) — при корректном учёте нулевой моды.

---

# 4) Итоговая формула для $\eta_j$ и сборка $T$

$$
\boxed{
\eta_j
=\ C^{(\mathrm{dir})}_j\ \cdot\ \Theta^{(\mathrm{pos})}_j,\qquad
C^{(\mathrm{dir})}_j=\exp\!\big(-\pi\,\delta\vec d^\dagger A^{(\mathrm{dir})}_j\,\delta\vec d\big).
}
$$

$$
\boxed{
T(z,\vec d)
=\sum_{j}\Big(\alpha_j\cdot\mathrm{Re}\,\eta_j\Big)\,\hat T_j.
}
$$

Все обозначения, блоки и коэффициенты совпадают с каноном; никаких новых объектов не добавлено.

---

# 5) Инварианты и условия корректности (для тестов)

1. **Тождественная эквивалентность** реальной и дуальной форм $\Theta^{(\mathrm{pos})}_j$ (одно и то же значение для любого $t>0$).
2. **Отсутствие решётки по dir**: $\eta_j$ всегда содержит только один гауссов множитель $C^{(\mathrm{dir})}_j$ без суммирования.
3. **Дискретные хвостовые оценки** (выше) — обеспечивают верхние границы остатка при усечении окон $L_{\mathrm{real}},L_{\mathrm{dual}}$.
4. **Линейность по $(\alpha_j,\hat T_j)$** и инвариантность к перестановкам/разбиениям фиктивных «пакетов» (это следствие линейности суммы).
5. **Согласие с каноном**: все шаги (iota, $R$, $\mathcal R$, $\Sigma$, $q$, $\rho$, периодизация только по позиции) полностью сохранены.

---

# 6) Краткий алгоритмический протокол

Для каждого $j$:

1. Построить $R(\vec d_j)$; задать $A^{(\mathrm{pos})}_j=A^{(\mathrm{dir})}_j=R\,\mathrm{diag}(\sigma_{\parallel j}^{-1},\sigma_{\perp j}^{-1},\ldots)R^\dagger$.
2. Вычислить $C^{(\mathrm{dir})}_j=\exp(-\pi\,\delta\vec d^\dagger A^{(\mathrm{dir})}_j\,\delta\vec d)$.
3. Задать $b_{\mathrm{pos}}=\delta z$ (в фундаментальной области тора).
4. Выбрать режим: **реальный** или **дуальный** (или гибридный Ewald) и параметр $t>0$.
5. Просуммировать $\Theta^{(\mathrm{pos})}_j$ в выбранной форме до окон $L_{\mathrm{real}}$ или $L_{\mathrm{dual}}$, подобранных из хвостовых оценок под допуск $\varepsilon$.
6. $\eta_j=C^{(\mathrm{dir})}_j\cdot \Theta^{(\mathrm{pos})}_j$.
7. Сложить $T=\sum_j (\alpha_j\,\mathrm{Re}\,\eta_j)\,\hat T_j$.
