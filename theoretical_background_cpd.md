# Change Point Detection in Irregularly Sampled Time Series — AGN Light Curves Toy Example

**Theoretical Background and Toy Example Description**

*Side document to the student presentation delivered at the XXI Escuela de Verano en Matemáticas Discretas (Discretas 2026), Universidad Adolfo Ibáñez, Viña del Mar, Chile.*

**Author:** Cinthya Leonor Vergara Silva · ORCID: [0000-0003-4107-6135](https://orcid.org/0000-0003-4107-6135) · 2026

*This event was funded by the Center for Mathematical Modeling (CMM) through its ANID FB210005 Basal Project.*

---

## Table of Contents

1. [Theoretical Background](#1-theoretical-background) (incl. method landscape)
2. [CPD in Astronomy and AGN Science](#2-cpd-in--astronomy-and-agn-science)
3. [Toy Example: Description and Workflow](#3-toy-example-description-and-workflow)
4. [Citation and Academic Notes](#4-citation-and-academic-notes)
5. [References](#5-references)

---

## 1. Theoretical Background

### 1.1 Motivation and Historical Context

The systematic detection of changes has accompanied human civilization since antiquity. Celestial observation was essential for the construction of calendars, the understanding of climatic seasons, maritime navigation, and the philosophical and religious development of societies (Daniel 1973; Frisinger 2018). Across biology, genetics, chemistry, physics, health, economics, social sciences, cybersecurity, and finance, the analysis of changes has been fundamental to the development of new theories and technologies.

**Change point detection (CPD)** is the methodological framework for identifying *when* and *how* the statistical properties of a stochastic process change over time. The field has evolved from early applications in industrial quality control into a rigorous discipline at the intersection of statistical inference, optimization theory, and applied mathematics (Aminikhanghahi & Cook 2017; Truong et al. 2020; Gupta et al. 2024).

### 1.2 Definitions and Taxonomy

Change points appear in the literature under several names — **thresholds, segmentation, structural breaks, regime switching, splitting points, breakpoints,** or **detecting disorder** — depending on the context and technique. These terms are largely interchangeable; the distinction lies primarily in the application domain or specific algorithm (Baseville et al. 1993; Brodsky & Darkhovsky 1993; Truong et al. 2020).

A useful taxonomy spans several **orthogonal** axes:

| Dimension | Categories |
|---|---|
| Number of changes | Single changepoint vs. multiple changepoints |
| K known a priori? | Fixed K vs. estimated K |
| Target of change | Mean, variance, correlation, distribution, regression structure, or combinations |
| Model parameters | Known (e.g. known variance) vs. unknown |
| Inference paradigm | Frequentist (penalized likelihood, hypothesis test) vs. Bayesian (posterior, MAP) |
| Model assumptions | Parametric vs. non-parametric vs. semi-parametric |
| Timing of analysis | Online (sequential, real-time) vs. offline (retrospective, batch) |
| Label availability | Supervised vs. unsupervised |

**Structural break detection** is a specific subtype in which the change is not merely a level shift but a more complex change in the structure or parameters of the underlying model — potentially including a change in the generating distribution itself. Each regime can be characterized by a distinct stochastic process with its own parameter vector (Bai & Perron 1998; Aue & Horváth 2013; Truong et al. 2020).

### 1.3 Formal Problem Statement

The canonical formulation follows Killick et al. (2012) and Truong et al. (2020).

> **Formal Definition.** Given a time series or indexed stochastic process $\{y_1, y_2, \ldots, y_T\}$, identify an unknown number $K$ of changepoints at ordered indices
>
> $$\tau_0 = 0 \;<\; \tau_1 \;<\; \tau_2 \;<\; \cdots \;<\; \tau_K \;<\; \tau_{K+1} = T$$
>
> such that observations in segment $(\tau_i, \tau_{i+1}]$ follow distribution $P_i$, with $P_i \neq P_{i+1}$ for each $i = 0, \ldots, K$. The $K$ changepoints define $\mathbf{K+1}$ **segments**. The problem simultaneously estimates: (i) the number of segments $K+1$, (ii) the changepoint locations $\{\tau_i\}_{i=1}^K$, and (iii) the segment-specific parameters or distributions $\{P_i\}_{i=0}^K$.

### 1.4 The Penalized Segmentation Objective

The most common computational approach formulates CPD as a **penalized optimization problem** (Lavielle 2005; Killick et al. 2012):

$$\min_{\{K,\,\tau\}} \left[ \sum_{i=0}^{K} \mathcal{C}\!\left(y_{\tau_i+1:\tau_{i+1}}\right) + \beta K \right]$$

where $\mathcal{C}(\cdot)$ is a segment cost function measuring within-segment homogeneity (lower = more homogeneous), $\beta > 0$ is a penalty controlling model complexity, and the sum runs over the $K+1$ segments.

**Common cost functions:**

- **L2 norm:** $\mathcal{C} = \sum_t \|y_t - \bar{y}\|^2$ — targets changes in mean.
- **L1 norm:** $\mathcal{C} = \sum_t |y_t - \bar{y}|$ — robust to outliers; targets changes in median.
- **Negative log-likelihood:** $\mathcal{C} = -\sum_t \log f(y_t \mid \hat{\theta})$ — parametric MLE approach; targets changes in mean and/or variance depending on the distributional assumption.
- **Kernel / MMD-based:** $\mathcal{C} = \frac{1}{n^2}\sum_{i,j} k(y_i, y_j) - \frac{2}{n}\sum_i k(y_i, \bar{y}_k)$ — non-parametric, distribution-free; measures within-segment discrepancy via a kernel function $k(\cdot,\cdot)$ without requiring a distributional assumption.

**Penalty selection.** For a model with $p$ free parameters per segment:

- **AIC:** $\beta = 2p$
- **BIC:** $\beta = p\log(T)$

> **Note on the toy example.** The `ruptures` library uses `pen=np.log(n)`, corresponding to BIC with $p = 1$. The `normal` cost model estimates both mean and variance per segment, so the theoretically correct count is $p = 2$; the choice $p = 1$ is a lighter penalty that permits more changepoints. This is a deliberate simplification for the teaching context, not a claim of optimality.

### 1.5 Bayesian Interpretation and MAP Equivalence

The frequentist penalized objective has a Bayesian interpretation (Wang et al. 2024). MAP estimation of the partition gives:

$$\underset{K,\,\tau}{\arg\max}\; p(K, \tau \mid y) \;=\; \underset{K,\,\tau}{\arg\min} \left[ -\log p(y \mid K, \tau) - \log p(K, \tau) \right]$$

Identifying $-\log p(y \mid K, \tau)$ with the cost $\sum \mathcal{C}(\cdot)$ and $-\log p(K, \tau)$ with the penalty $\beta K$ — under a geometric or Poisson prior on $K$ and a uniform prior on locations given $K$ — recovers exactly the frequentist penalized objective. The likelihood $p(y \mid K, \tau)$ defines the cost function; the prior $p(K, \tau) = p(K) \cdot p(\tau \mid K)$ provides regularization. This equivalence justifies interpreting $\beta$ as encoding prior beliefs about the expected number of changepoints.

### 1.6 PELT: Pruned Exact Linear Time

PELT (Killick et al. 2012) solves the penalized segmentation objective **exactly** with $O(n)$ expected computational cost under the assumption that the number of true changepoints grows linearly with $n$. It is a dynamic programming algorithm augmented by a pruning rule.

Let $F(t)$ denote the optimal (minimum) cost of segmenting $y_{1:t}$. PELT computes:

$$F(t) = \min_{\tau < t} \left[ F(\tau) + \mathcal{C}(y_{\tau+1:t}) + \beta \right]$$

with boundary condition $F(0) = -\beta$. The admissible set $\mathcal{R}_t$ of candidate last-changepoints is initialized as $\mathcal{R}_1 = \{0\}$.

**Pruning rule (Killick et al. 2012, Theorem 3.1).** Candidate $\tau$ is permanently removed from $\mathcal{R}_t$ if:

$$F(\tau) + \mathcal{C}(y_{\tau+1:t}) + K \;\leq\; F(t)$$

where $K$ is a constant satisfying $\mathcal{C}(y_{a:b}) \geq \mathcal{C}(y_{a:m}) + \mathcal{C}(y_{m+1:b}) + K$ for all valid $a \leq m < b$ (for L2 cost, $K = 0$; for negative Gaussian log-likelihood, $K = -\beta$).

> **Direction of the inequality.** The condition removes $\tau$ when $F(\tau) + \mathcal{C}(\cdot) + K \leq F(t)$, i.e. when the candidate is already at least as costly as the current optimum at $t$. Candidates that are strictly cheaper are **retained** — they might be optimal for a future split at $t' > t$. The direction is $\leq F(t)$, not $\geq$.

PELT finds the **globally optimal** partition for the given cost and penalty $\beta$. It is not a greedy approximation. Its $O(n)$ expected complexity (vs $O(n^2)$ for unconstrained dynamic programming) makes it practical for series of hundreds to tens of thousands of observations.

### 1.7 Optimization Strategies by Problem Type

| Setting | Representative algorithms |
|---|---|
| K unknown — exact | PELT (Killick et al. 2012): $O(n)$ expected; Optimal Partitioning / DP: $O(n^2)$ |
| K unknown — approximate | Binary Segmentation (BS): greedy; Wild BS; SMUCE |
| K known — exact | Optimal Partitioning with fixed K |
| K known — approximate | Window-based methods; Bottom-Up Segmentation |
| Online / sequential | CUSUM (Page 1954); BOCD (Adams & MacKay 2007); LSCUSUM |
| Bayesian offline | BCPA; reversible-jump MCMC; Bayesian Blocks (Scargle et al. 2013) |

> **Note on Binary Segmentation.** BS is naturally framed for known $K$ or with a separate stopping criterion. It does *not* minimize the global penalized objective and can produce suboptimal partitions when changepoints interact.

> **Note on CUSUM.** CUSUM is an *online/sequential* method — it belongs to the online–offline axis of the taxonomy, not to the supervised–unsupervised axis. Classifying it under "unsupervised methods" conflates two independent classification dimensions.

### 1.8 Historical Development

#### 1.8.1 Origins: Statistical Quality Control

The conceptual foundations of CPD trace to industrial mass production. Shewhart (1931) established the scientific framework for statistical quality control through Specification, Production, and Inspection, using empirical three-sigma decision rules to detect assignable causes of variability.

#### 1.8.2 Mathematical Formalization (1940s–1960s)

- **Wald (1947):** sequential testing; definitions of delay and Type I error control.
- **Girshick & Rubin (1952):** Bayesian formulation of sequential detection.
- **Page (1954, 1961):** CUSUM chart — motivated by the likelihood ratio — targeting single location-parameter shifts.
- **Lorden (1971)** and **Moustakides (1986)** gave optimality properties of CUSUM under minimax criteria (see surveys in Truong et al. 2020; Horváth & Rice 2024).

#### 1.8.3 Methodological Evolution Axes

The field has evolved along five orthogonal axes (Truong et al. 2020; Gupta et al. 2024):

1. **Parametric → Non-parametric:** relaxation of distributional assumptions.
2. **Offline → Online:** retrospective batch to real-time sequential inference.
3. **Low-dimensional → High-dimensional:** scalability to $p \gg n$.
4. **Univariate → Complex data:** multivariate series, networks, functional, spatial.
5. **Single type → Multiple types:** joint detection of changes in mean, variance, and distribution.

### 1.9 Fundamental Trade-offs

- **Bias–Variance:** parametric methods have higher power under correct specification; non-parametric methods are more robust.
- **Computation–Accuracy:** greedy methods are fast but suboptimal; exact methods (PELT, DP) are optimal but heavier.
- **Sensitivity–Specificity:** penalty $\beta$ directly controls the detection power vs. false alarm trade-off.
- **Local–Global:** sequential methods cannot revise earlier decisions; batch methods optimize globally.

### 1.10 Current Open Challenges

- **Data-related:** irregular sampling, temporal dependence, low SNR, seasonal gaps, graph-structured and functional data.
- **Computational complexity:** scalability to high-dimensional or massive datasets (Wang et al. 2024; Lee et al. 2025).
- **Sensitivity–specificity:** rigorous false alarm control at scale (Aminikhanghahi & Cook 2017; Horváth & Rice 2024).
- **Online vs. offline:** combining sequential detection with global optimality.
- **Robustness:** outliers, missing data, distributional misspecification, non-stationarity.
- **Statistical guarantees:** consistency, power, scalability, provable computational bounds (Chakrabarty et al. 2025).
- **Causality and interpretability:** causal mechanisms behind detected changepoints; explaining deep learning outputs (Assaad et al. 2022; Yuan et al. 2025).

### 1.11 Literature and Method Landscape (Summary)

CPD methods are often grouped by setting: **offline unsupervised** (likelihood-based, density-ratio, Bayesian, kernel/graph), **online/sequential** (e.g. CUSUM, BOCD), **segmentation-based** (Binary Segmentation, Bottom-Up, Window-Based, PELT), **supervised** (when labels exist), and **Bayesian** (BCPA, BOCD, reversible-jump MCMC). In practice, CPD is used across healthcare, biology, neuroscience, climatology, economics, finance, geophysics, oceanography, cybersecurity, industrial monitoring, and astronomy; see Section 2 for astronomical applications and the References for key surveys (Baseville et al. 1993; Truong et al. 2020; Gupta et al. 2024; Horváth & Rice 2024).

---

## 2. CPD in Astronomy and AGN Science

### 2.1 Astronomical Applications

CPD is applied across a broad range of astronomical challenges (Scargle 1998; Aminikhanghahi & Cook 2017; Ting et al. 2025):

- Segmentation of astronomical time series and astrophysical images (Scargle 1998; Xu et al. 2021).
- Detection of transient phenomena: flares, eclipses, planetary transits (Kim & Bailer-Jones 2009; Graham et al. 2023; He et al. 2025).
- Correction of radial velocity measurements for stellar activity (Delisle et al. 2022).
- Separation of mixed physical processes, e.g. AGN vs. transients (Sharma et al. 2024).
- Detection of AGN state transitions and nuclear variability events, including changing-look phenomena and candidate turn-on AGN (Sánchez-Sáez et al., 2024).
- Operational management of large survey datasets (Das et al. 2009).

Common challenges: **irregular temporal sampling**, **data heterogeneity** across instruments, **seasonal gaps**, low SNR, and rigorous false positive control over millions of objects (Baseville et al. 1993; Horváth & Rice 2024; Jin et al. 2025).

### 2.2 Active Galactic Nuclei

AGN are regions surrounding supermassive black holes (SMBHs) where infalling matter forms an **accretion disk**. Brightness varies due to changes in accretion rate, disk instabilities, relativistic effects, or stochastic red noise.

Changing-look AGN (CL-AGN) are a well-established observational class defined by a spectroscopic transition between AGN types, in which broad emission lines appear or disappear. These transitions are commonly interpreted as changes in accretion rate or ionising flux, although alternative explanations (e.g., variable obscuration) may also apply

A related but physically different phenomenon is that of a **turn-on AGN**: a previously *completely dormant* nucleus — showing no prior AGN activity over decades of archival observations — that begins accreting and brightening persistently. Unlike CL-AGN (where a known active nucleus changes spectral type), a turn-on event originates from a genuinely quiescent host galaxy. The informal term "awakening" is used colloquially for such events but is not a formal established classification. The physical origin of turn-on events is still debated: they may represent genuine AGN activation, unusual tidal disruption events (TDEs), or a still-unknown class of nuclear transients.

### 2.3 The ZTF19acnskyy Event

| Property | Value |
|---|---|
| Object name | ZTF19acnskyy |
| SDSS designation | SDSS J133519.91+072807.4 |
| Coordinates (ICRS) | RA = 13h 35m 19.91s, Dec = +07d 28m 07.4s |
| Event date | 13 December 2019 |
| Black hole mass | ~$10^6\,M_\odot$ |
| Data source | ZTF g-band, public IRSA archive |
| Observations | ~153 points in g-band |
| Reference | Sánchez-Sáez et al. (2024) |

---

## 3. Toy Example: Description and Workflow

### 3.1 Purpose and Scope

A **minimal, reproducible teaching demonstration** of CPD on a real astronomical light curve. Not a production pipeline or validated scientific analysis. Goals:

- Demonstrate programmatic ZTF data access via Astropy and IRSA.
- Illustrate the three canonical CPD types: change in mean, variance, both simultaneously.
- Apply PELT in practice with a real, irregularly sampled light curve.
- Fit a piecewise parametric model to each detected regime.
- Produce clear, reproducible figures for a teaching presentation.

### 3.2 Dependencies and Setup

```bash
pip install -r requirements-astropy.txt
```

| Package | Role |
|---|---|
| `astropy` | `SkyCoord`, `Table.read` (IPAC), `TimeSeries`, `Time`, units |
| `numpy` | Numerical operations, design matrices, OLS |
| `matplotlib` | All figure generation |
| `ruptures` | PELT (Truong et al. 2020) |

### 3.3 Data Acquisition from IRSA

```python
COORD = SkyCoord('13h35m19.91s', '+07d28m07.4s', frame='icrs')
params = urlencode({
    'CIRCLE': f'{COORD.ra.deg} {COORD.dec.deg} {5/3600}',
    'BANDNAME': 'g',
    'BAD_CATFLAGS_MASK': 32768,   # exclude poor-quality ZTF observations
    'FORMAT': 'ipac_table',
})
tbl = Table.read(io.BytesIO(response), format='ipac')
```

Data are sorted by MJD; a sequential integer index `t` is created. **PELT operates on this integer index, not on MJD directly** — see Section 3.7.

### 3.4 Figures Produced

#### Figure 1 — `change-point-types.png`
Three simulated light curves (seed 42) illustrating change in mean only, variance only, and both simultaneously. Boundaries at indices 100, 250, 450.

#### Figure 2 — `change-in-mean.png`
PELT on the raw magnitude series (L2 cost; lighter penalty so mean shifts are detected). Segment means plotted per regime.

#### Figure 3 — `change-in-variance.png`
PELT on **mean-subtracted** series (targets variance changes). `normal` cost, `pen=np.log(n)`.

```python
algo = rpt.Pelt(model='normal', min_size=3)
algo.fit((mag - np.mean(mag)).reshape(-1, 1))  # centred: targets variance
cpts = algo.predict(pen=np.log(n))
```

#### Figure 4 — `change-in-mean-variance.png`
PELT on **raw** magnitude series (targets simultaneous mean + variance changes). Same cost and penalty, non-centred data. Comparing with Figure 3 illustrates how the signal representation affects detected changepoints.

#### Figure 5 — `piecewise-models.png`
Left: piecewise parametric fits per segment (changepoints from binary segmentation, adaptive-pooled). Right: coefficient comparison across segments. The implementation is close in spirit to least-squares estimation of a shift in linear processes (Bai 1994).

The model fitted independently per segment:

$$
y(t) = \alpha + \beta t + \gamma t^2 + a_1 \cos(2\pi f_1 t) + b_1 \sin(2\pi f_1 t) + \varepsilon
$$

where $f_1 = 1/182.6\,\mathrm{days}$ converted to index units via `p_index = 182.6 / np.mean(np.diff(mjd))`. Coefficients estimated by OLS (`numpy.linalg.lstsq`) per segment. The 182.6-day period is illustrative; AGN variability is not strictly periodic (Scargle et al. 2013).

### 3.5 Running the Example

```bash
python scripts/plots_astropy.py

```


### 3.7 Limitations (By Design)

- **No domain validation:** changepoints not compared to known event date (13 Dec 2019) or published break points.
- **PELT on index:** PELT uses integer index `t`, not MJD. Methods such as Bayesian Blocks (Scargle et al. 2013) handle irregular sampling natively.
- **Penalty not calibrated:** `log(n)` for variance/mean–variance panels; lighter penalty for mean-only panel; not optimized for this dataset.
- **Generic piecewise model:** binary segmentation + trend + harmonic per segment; does not reflect domain-specific AGN variability models (e.g. damped random walk, CARMA).
- **No uncertainty quantification:** no confidence intervals on changepoint locations.

---

## 4. Citation and Academic Notes

```
Vergara Silva, C. L. (2026). Change Point Detection in Irregularly Sampled Time Series — AGN Light Curves Toy Example. Student Talk Summer School in Discrete Mathematics , CMM, January 2026, UAI, Viña del Mar, Chile.
Zenodo. https://doi.org/10.5281/zenodo.18779915
```


---

## 5. References


**Adams, R.P. & MacKay, D.J.C. (2007).** Bayesian online changepoint detection. *arXiv preprint* arXiv:0710.3742.

**Aminikhanghahi, S. & Cook, D.J. (2017).** A survey of methods for time series change point detection. *Knowledge and Information Systems*, 51(2), 339–367.

**Assaad, C.K., Devijver, E., & Gaussier, E. (2022).** Survey and evaluation of causal discovery methods for time series. *Journal of Artificial Intelligence Research*, 73, 767–819.

**Aue, A. & Horváth, L. (2013).** Structural breaks in time series. *Journal of Time Series Analysis*, 34(1), 1–16.

**Bai, J. (1994).** Least squares estimation of a shift in linear processes. *Journal of Time Series Analysis*, 15(5), 453–472.

**Bai, J. & Perron, P. (1998).** Estimating and testing linear models with multiple structural changes. *Econometrica*, 66(1), 47–78.

**Baseville, V., Plante, R., Deeming, T., & Durand, D. (1993).** Change point analysis in astronomy. *Astronomy and Astrophysics*, 271, 451.

**Brodsky, E. & Darkhovsky, B.S. (1993).** *Nonparametric Methods in Change Point Problems*. Springer.

**Chakrabarty, S., Chakraborty, A., & De, S.K. (2025).** Consistent detection and estimation of multiple structural changes in functional data. *arXiv preprint* arXiv:2511.14353.

**Daniel, H. & International Meteorological Organization (1973).** *One Hundred Years of International Co-operation in Meteorology (1873–1973): A Historical Review*. WMO.

**Das, K., Bhaduri, K., Arora, S., et al. (2009).** Scalable distributed change detection from astronomy data streams using local, asynchronous eigen monitoring algorithms. *Proc. SIAM International Conference on Data Mining (SDM)*, 247–258.

**Delisle, J.-B., Hara, N., & Ségransan, D. (2022).** Accounting for stellar activity signals in radial-velocity data by using change point detection techniques. *Astronomy & Astrophysics*, 659, A182.

**Fryzlewicz, P. (2014).** Wild binary segmentation for multiple change-point detection. *The Annals of Statistics*, 42(6), 2243–2281.

**Frisinger, H.H. (2018).** *History of Meteorology to 1800*. Springer.

**Girshick, M.A. & Rubin, H. (1952).** A Bayes approach to a quality control model. *The Annals of Mathematical Statistics*, 23(1), 114–125.

**Graham, M.J., McKernan, B., Ford, K.E.S., et al. (2023).** A light in the dark: searching for electromagnetic counterparts to black hole–black hole mergers in LIGO/Virgo O3 with the Zwicky Transient Facility. *The Astrophysical Journal*, 942(2), 99.

**Gupta, M., Wadhvani, R., & Rasool, A. (2024).** Comprehensive analysis of change-point dynamics detection in time series data: A review. *Expert Systems with Applications*, 248, 123342.

**He, L., Liu, Z.-Y., Niu, R., et al. (2025).** A systematic search for AGN flares in ZTF Data Release 23. *arXiv preprint* arXiv:2507.20232.

**Horváth, L. & Rice, G. (2024).** *Change Point Analysis for Time Series*. Springer.

**Jin, B. & Li, J. (2025).** *Change Point Analysis: Theory and Application*. CRC Press.

**Killick, R., Fearnhead, P., & Eckley, I.A. (2012).** Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590–1598.

**Kim, D.-W., Protopapas, P., Alcock, C., et al. (2009).** Variability detection by change-point analysis. *Monthly Notices of the Royal Astronomical Society*, 397(1), 558–568.

**Lavielle, M. (2005).** Using penalized contrasts for the change-point problem. *Signal Processing*, 85(8), 1501–1510.

**Lee, S., An, J., Mikhaylov, A., et al. (2025).** Linearly penalized segmentation, binary segmentation, bottom-up segmentation and window-based methods for change point detection. *Discover Artificial Intelligence*, Springer.

**Page, E.S. (1954).** Continuous inspection schemes. *Biometrika*, 41(1/2), 100–115.

**Page, E.S. (1961).** Cumulative sum charts. *Technometrics*, 3(1), 1–9.

**Sánchez-Sáez, P., Hernández-García, L., Bernal, S., et al. (2024).** SDSS1335+0728: The awakening of a ~10⁶ M⊙ black hole. *Astronomy & Astrophysics*, 688, A157.

**Scargle, J.D. (1998).** Studies in astronomical time series analysis. V. Bayesian blocks, a new method to analyze structure in photon counting data. *The Astrophysical Journal*, 504(1), 405–418.

**Scargle, J.D., Norris, J.P., Jackson, B., & Chiang, J. (2013).** Studies in Astronomical Time Series Analysis. VI. Bayesian Block Representations. *The Astrophysical Journal*, 764(2), 167.

**Sharma, V., Trotta, R., & Messenger, C. (2024).** State space modelling for detecting and characterising gravitational waves afterglows. *Astronomy and Computing*, 48, 100751.

**Shewhart, W.A. (1931).** *Economic Control of Quality of Manufactured Product*. D. Van Nostrand, New York.

**Ting, Y.-S. (2025).** Statistical machine learning for astronomy — a textbook. *arXiv preprint* arXiv:2506.12230.

**Truong, C., Oudre, L., & Vayatis, N. (2020).** Selective review of offline change point detection methods. *Signal Processing*, 167, 107299.

**Wald, A. (1947).** *Sequential Analysis*. Courier Corporation.

**Wang, H. & Xie, Y. (2024).** Sequential change-point detection: Computation versus statistical performance. *Wiley Interdisciplinary Reviews: Computational Statistics*, 16(1), e1628.

**Xu, Y., Stein, N.M., Feigelson, E.D., & Babu, G.J. (2021).** Change-point detection and image segmentation for time series of astrophysical images. *The Astronomical Journal*, 161(4), 184.

**Xu, R., Song, Z., Wu, J., Wang, C., & Zhou, S. (2025).** Change-point detection with deep learning: A review. *Frontiers of Engineering Management*, 12(1), 154–176.

**Yuan, X., Wang, X., Kato, M., et al. (2025).** CiiNet: Self-iterative performance optimization for dynamic networks based on causal inference and interpretable evaluation. *IEEE Transactions on Networking*, 34, 1554–1568.

---

**Acknowledgments.** This event was funded by the Center for Mathematical Modeling (CMM) through its ANID FB210005 Basal Project.

*Author: Cinthya Leonor Vergara Silva · ORCID: 0000-0003-4107-6135*  
*Presentation theme: OsloMet Beamer
