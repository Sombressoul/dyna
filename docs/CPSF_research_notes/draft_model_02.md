**General Continuous Projection Memory (GCPM): Аналитически решаемая модель памяти по сумме направленных вкладов**

---

## Структура памяти

Память представляет собой список направленных объёмных вкладов:

\[\mathcal{M} = \{ (\vec{o}_j, \vec{d}_j, T_j, \Sigma_j) \}_{j=1}^N\]

- \(\vec{o}_j \in \mathbb{R}^D\) — центр вклада (origin)
- \(\vec{d}_j \in \mathbb{R}^D\), \(\|\vec{d}_j\| = 1\) — направление проекции
- \(T_j \in \mathbb{C}^C\) — вектор значения
- \(\Sigma_j \in \mathbb{R}^{2D \times 2D}\) — ковариация в пространстве \(\ell = (\vec{o}, \vec{d})\)

---

## READ: Чтение проекции

Выход проекции \(T(\ell)\) получается как сумма вкладов:

\[T(\ell) = \sum\limits_{j=1}^N T_j \cdot \psi_j(\ell)\]

Где:

\[\psi_j(\ell) = \mathcal{N}(\ell \mid \ell_j, \Sigma_j)\]

---

## CREATE: Запись

Добавление нового вклада:

\[\mathcal{M} \leftarrow \mathcal{M} \cup \left\{ (\vec{o}_{\text{new}}, \vec{d}_{\text{new}}, T^*, \Sigma_{\text{new}}) \right\}\]

---

## UPDATE: Коррекция смыслов

При \(T(\ell^*) = \sum_j T_j \cdot \psi_j(\ell^*)\), для желаемого \(T^*\):

\[T_j \leftarrow \frac{\psi_j(\ell^*)}{\sum_k \psi_k(\ell^*)} \cdot T^*\quad \text{(для локальной коррекции)}\]

---

## DELETE: Забывание

- \(T_j \leftarrow 0\)  — удаление
- Или \(\Sigma_j \to \infty\cdot I\) — распыление вклада

---

## FIND: Обратный поиск проекции

Для данного \(T^*\) ищется:

\[\ell^* = \frac{\sum_j \ell_j \cdot \exp\left( -\frac{\|T_j - T^*\|^2}{2\tau^2} \right)}{\sum_j \exp\left( -\frac{\|T_j - T^*\|^2}{2\tau^2} \right)}\]

---

## Дифференцируемость

Все функции:
- \(T(\ell)\), \(\psi_j(\ell)\)
- \(\ell^*(T^*)\)

гладкие и дифференцируемы по всем параметрам:

- \(T_j\), \(\vec{o}_j\), \(\vec{d}_j\), \(\Sigma_j\), \(\tau\), \(T^*\)

---

## Вывод

Эта модель GCPM:
- Полностью сохраняет параметрическую выразительность
- Все CRUD-операции аналитически решаемы
- Вся система гладка и обучаема

