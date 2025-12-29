### **Fundamentals of Inference**
* **Probability**
    *   **Probability Spaces:** Triple $(\Omega, \mathcal{A}, P)$ where $\Omega$ is the sample space, $\mathcal{A}$ is a $\sigma$-algebra, and $P$ is a probability measure.
        *   **Kolmogorov Axioms:** 
            *   $0 \le P(A) \le 1$
            *   $P(\Omega) = 1$
            *   For disjoint events $P(\bigcup_{i=1}^\infty A_i) = \sum_{i=1}^\infty P(A_i)$
    *   **Random Variables:** Function $X: \Omega \to T$. Probability $P(X \in S) = P(\{\omega \in \Omega : X(\omega) \in S\})$.
    *   **Distributions:**
        *   **PMF (Discrete):** $p_X(x) = P(X = x)$
        *   **CDF:** $P_X(x) = P(X \le x)$
    *   **Continuous Distributions:** Defined by PDF $p_X$ s.t. mass $m(M) = \int_M p_X(x) dx$.
        *   **Univariate Normal:** $N(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( - \frac{(x-\mu)^2}{2\sigma^2} \right)$
    *   **Joint Probability:** $P(A, B) = P(A \cap B)$
        *   **Sum Rule (Marginalization):** $p(x_{1:i-1}, x_{i+1:n}) = \int p(x_{1:n}) dx_i$
    *   **Conditional Probability:**
        *   **Definition:** $P(A | B) = \frac{P(A, B)}{P(B)}$
        *   **Product/Chain Rule:** $p(x_{1:n}) = p(x_1) \prod_{i=2}^n p(x_i | x_{1:i-1})$
        *   **Law of Total Probability (LOTP):** $p(x) = \int p(x | y) p(y) dy$
    *   **Independence:**
        *   **Independent ($X \perp Y$):** $p(x, y) = p(x)p(y)$
        *   **Conditionally Independent ($X \perp Y | Z$):** $p(x, y | z) = p(x | z)p(y | z)$
    *   ** Directed Graphical Models:** Generative model factorization: $p(x_{1:n}) = \prod_{i=1}^n p(x_i | \text{parents}(x_i))$.
    *   ** Expectation:** $E[X] = \int x \cdot p(x) dx$.
        *   **Linearity:** $E[AX + b] = AE[X] + b$.
        *   **LOTUS:** $E[g(X)] = \int g(x) p(x) dx$.
        *   **Tower Rule (LOTE):** $E[X] = E_Y[E_X[X | Y]]$.
    *   **Covariance and Variance:**
        *   **Covariance:** $\text{Cov}[X, Y] = E[(X-E[X])(Y-E[Y])^\top] = E[XY^\top] - E[X]E[Y]^\top$.
        *   **Variance:** $\text{Var}[X] = \text{Cov}[X, X]$.
        *   **Properties:** $\text{Var}[AX+b] = A\text{Var}[X]A^\top$ and $\text{Var}[X+Y] = \text{Var}[X] + \text{Var}[Y] + 2\text{Cov}[X, Y]$.
        *   **Law of Total Variance (LOTV):** $\text{Var}[X] = E_Y[\text{Var}_X[X | Y]] + \text{Var}_Y[E_X[X | Y]]$.
    *   **Change of Variables:** For $Y = g(X)$, $p_Y(y) = p_X(g^{-1}(y)) \cdot |\det(D g^{-1}(y))|$.
* **Probabilistic Inference**
    *   **Bayes Rule:** $p(x | y) = \frac{p(y | x) p(x)}{p(y)}$ where $p(x | y)$ is the posterior, $p(x)$ the prior, $p(y | x)$ the likelihood, and $p(y)$ the marginal likelihood/normalizing constant.
    *   **Maximum Entropy Principle:** Select prior with least additional assumptions by maximizing $H[p] = E_{x \sim p}[-\log p(x)]$.
    *   **Normal Distribution Inference:** For Gaussian random vector $X \sim N(\mu, \Sigma)$:
        *   **Multivariate PDF:** $N(x; \mu, \Sigma) = \frac{1}{\sqrt{\det(2\pi\Sigma)}} \exp \left( -\frac{1}{2} (x-\mu)^\top \Sigma^{-1} (x-\mu) \right)$.
        *   **Conditioning:** $X_A | X_B = x_B \sim N(\mu_{A|B}, \Sigma_{A|B})$ with:
            *   $\mu_{A|B} = \mu_A + \Sigma_{AB}\Sigma_{BB}^{-1}(x_B - \mu_B)$
            *   $\Sigma_{A|B} = \Sigma_{AA} - \Sigma_{AB}\Sigma_{BB}^{-1}\Sigma_{BA}$
* **Supervised Learning**
    *   **Model:** $y_i = f_\theta(x_i) + \epsilon_i$.
    *   **Maximum Likelihood (MLE):** $\hat{\theta}_{MLE} = \text{arg max}_\theta \sum_{i=1}^n \log p(y_i | x_i, \theta)$.
    *   **Maximum A Posteriori (MAP):** $\hat{\theta}_{MAP} = \text{arg min}_\theta [-\log p(\theta) + \ell_{nll}(\theta; D_n)]$ where $\log p(\theta)$ acts as a regularizer.
    *   **Predictive Posterior:** $p(y^* | x^*, D_n) = \int p(y^* | x^*, \theta) p(\theta | D_n) d\theta$.
    *   **Credible Set:** $P(y^* \in C_\delta(x^*) | x^*, D_n) \ge 1 - \delta$.
    *   **Recursive Update:** $p^{(t+1)}(\theta) \propto p^{(t)}(\theta) \cdot p(y_{t+1} | \theta)$.
* **Outlook: Decision Theory**
    *   **Optimal Decision Rule:** $a^*(x) = \text{arg max}_{a \in A} E_{y|x}[r(y, a)]$.
    *   **Squared Loss:** Optimal action is the mean $a^*(x) = E[y | x]$.
    *   **Asymmetric Loss:** Penalizes under/overestimation differently; for Gaussian $y|x$, $a^*(x) = \mu_x + \sigma_x \cdot \Phi^{-1} \left( \frac{c_1}{c_1 + c_2} \right)$
### **Linear Regression**
* **Setup and Estimators**
    *   **Model Class:** Linear functions of the form $f(x; w) = w^\top x$.
    *   **Design Matrix ($X$):** Collection of inputs $x_1 \dots x_n$ where $X \in \mathbb{R}^{n \times d}$.
    *   **Least Squares Estimator:** $\hat{w}_{ls} = (X^\top X)^{-1}X^\top y$.
    *   **Ridge Regression:** $\hat{w}_{ridge} = (X^\top X + \lambda I)^{-1}X^\top y$.
    *   **Maximum Likelihood (MLE):**
    *   **Weights:** $\hat{w}_{MLE} = \hat{w}_{ls}$ under homoscedastic Gaussian noise.
    *   **Noise Variance:** $\hat{\sigma}^2_n = \frac{1}{n} \sum_{i=1}^n (y_i - w^\top x_i)^2$.

* **Weight-space View (Bayesian Linear Regression)**
    *   **Generative Model:**
        *   **Prior:** $w \sim N(0, \sigma^2_p I)$.
        *   **Likelihood:** $y_i | x_i, w \sim N(w^\top x_i, \sigma^2_n)$.
        *   **Posterior Distribution:** $w | x_{1:n}, y_{1:n} \sim N(\mu, \Sigma)$.
        *   **Mean:** $\mu = \sigma^{-2}_n \Sigma X^\top y$.
        *   **Covariance:** $\Sigma = (\sigma^{-2}_n X^\top X + \sigma^{-2}_p I)^{-1}$.
    *   **MAP Estimate:** $\hat{w}_{MAP} = \mu$ (equivalent to **Ridge Regression** with weight decay $\lambda = \sigma^2_n / \sigma^2_p$).
    *   **Predictive Posterior:**
        *   **Latent Function Value ($f^*$):** $f^* | x^*, D \sim N(\mu^\top x^*, x^{*\top} \Sigma x^*)$.
    *   **Observation Label ($y^*$):** $y^* | x^*, D \sim N(\mu^\top x^*, x^{*\top} \Sigma x^* + \sigma^2_n)$.

* **Aleatoric and Epistemic Uncertainty**
    *   **Law of Total Variance Decomposition:** $Var[y^* | x^*] = \mathbb{E}_\theta [Var_{y^*} [y^* | x^*, \theta]] + Var_\theta [\mathbb{E}_{y^*} [y^* | x^*, \theta]]$.
    *   **Aleatoric Uncertainty:** $\sigma^2_n$ (irreducible noise average across models).
    *   **Epistemic Uncertainty:** $x^{*\top} \Sigma x^*$ (variability of mean prediction due to lack of data).

* **Non-linear Regression**
    *   **Feature Transformation:** Apply $\phi : \mathbb{R}^d \to \mathbb{R}^e$ to inputs $x_i$.
    *   **Transformed Design Matrix:** $\Phi \in \mathbb{R}^{n \times e}$.
    *   **Polynomial Regression:** $\phi(x) = [1, x_1, \dots, x_d, x^2_1, \dots]$ results in dimension $e = \Theta(d^m)$.

* **Function-space View**
    *   **Prior on Function Values:** $f | X \sim N(0, K)$.
    *   **Kernel Matrix ($K$):** $K = \sigma^2_p \Phi \Phi^\top$ where $K_{ij} = \sigma^2_p \phi(x_i)^\top \phi(x_j)$.
    *   **Kernel Function:** $k(x, x') = \sigma^2_p \phi(x)^\top \phi(x') = Cov[f(x), f(x')]$.
    *   **The Kernel Trick:** Implicitly determines the class of functions $f$ is sampled from without explicit feature computation.
    *   **Polynomial Kernel:** $(1 + x^\top x')^m \approx \phi(x)^\top \phi(x')$.

### **Filtering**
* **Conditioning and Prediction**
    * **State Space Models**
        *   **Goal:** Keep track of a sequence of **hidden states** $(X_t)_{t\in\mathbb{N}_0}$ in $\mathbb{R}^d$ based on **noisy observations** $(Y_t)_{t\in\mathbb{N}_0}$ in $\mathbb{R}^m$.
        *   **Markov Property:** Future behavior is independent of the past given the present state: $X_{t+1} \perp X_{1:t-1}, Y_{1:t-1} | X_t$.

    * **Conditioning and Prediction**
        *   **Recursive Scheme:** Filtering alternates between two phases to update the agent's belief $b_t(x) = P(X_t = x | y_{1:t})$.
    *   **Conditioning (Update):** Incorporates a new observation $y_t$.
        *   $p(x_t | y_{1:t}) = \frac{1}{Z} p(x_t | y_{1:t-1}) p(y_t | x_t)$.
    *   **Prediction:** Forecasts the next state.
        *   $p(x_{t+1} | y_{1:t}) = \int p(x_{t+1} | x_t) p(x_t | y_{1:t}) dx_t$.
    *   **Bayesian Smoothing:** Estimating $X_k$ given data *beyond* time $k$ (i.e., $X_k | y_{1:t}$ for $t > k$) using a **forward-backward algorithm**.

* **Kalman Filters**
    *   **Definition:** A Bayes filter where the distribution over states is **Gaussian** and transitions are **conditional linear Gaussians**.
    *   **Models:**
        *   **Prior:** $X_0 \sim N(\mu, \Sigma)$.
        *   **Motion Model (Transition):** $X_{t+1} = FX_t + \epsilon_t$, where $\epsilon_t \sim N(0, \Sigma_x)$.
        *   **Sensor Model (Observation):** $Y_t = HX_t + \eta_t$, where $\eta_t \sim N(0, \Sigma_y)$
    *   **Note:** Online Bayesian Linear Regression is a special case of a Kalman filter where the hidden state (weights $w$) is constant ($F=I, \epsilon=0$).

    * **Kalman Update (Conditioning)**
        *   Given the prior belief $X_t | y_{1:t} \sim N(\mu_t, \Sigma_t)$, the updated belief is $X_{t+1} | y_{1:t+1} \sim N(\mu_{t+1}, \Sigma_{t+1})$.
        *   **Updated Mean:** $\mu_{t+1} = F\mu_t + K_{t+1}(y_{t+1} - HF\mu_t)$.
        *   **Updated Covariance:** $\Sigma_{t+1} = (I - K_{t+1}H)(F\Sigma_tF^\top + \Sigma_x)$.
    *   **Kalman Gain ($K_{t+1}$):** Determines how much weight to give the new observation.
        *   $K_{t+1} = (F\Sigma_tF^\top + \Sigma_x)H^\top (H(F\Sigma_tF^\top + \Sigma_x)H^\top + \Sigma_y)^{-1}$.
    *   **Error Term:** $(y_{t+1} - HF\mu_t)$ is the difference between the actual and predicted observation.

    * **Predicting**
        *   The marginal posterior of the next state before seeing the observation $y_{t+1}$ is $X_{t+1} | y_{1:t} \sim N(\hat{\mu}_{t+1}, \hat{\Sigma}_{t+1})$.
        *   **Prediction Mean:** $\hat{\mu}_{t+1} = F\mu_t$.
        *   **Prediction Covariance:** $\hat{\Sigma}_{t+1} = F\Sigma_tF^\top + \Sigma_x$.

### **Gaussian Processes**
* **Definition and Characterization**
    *   **Definition:** A **Gaussian Process (GP)** is an infinite set of random variables such that any finite number of them are **jointly Gaussian**.
    *   **Interpretation:** A GP is a normal distribution over functions, often called an **"infinite-dimensional Gaussian"**.
    *   **Characterization:** Defined by a **mean function** $\mu(x)$ and a **covariance (kernel) function** $k(x, x')$:
        *   For any set of points $A = \{x_1, \dots, x_m\}$, $f_A \sim N(\mu_A, K_{AA})$.
    *   **Observation model:** $y_* | x_* \sim N(\mu(x_*), k(x_*, x_*) + \sigma^2_n)$.

* **Learning and Inference**
    *   **Joint Distribution:** The observations $y_{1:n}$ and the noise-free prediction $f_*$ are jointly Gaussian.
    *   **Posterior Distribution:** $f | x_{1:n}, y_{1:n} \sim GP(\mu', k')$ where:
        *   **Posterior Mean:** $\mu'(x) = \mu(x) + k^\top_{x,A}(K_{AA} + \sigma^2_n I)^{-1}(y_A - \mu_A)$.
    *   **Posterior Covariance:** $k'(x, x') = k(x, x') - k^\top_{x,A}(K_{AA} + \sigma^2_n I)^{-1}k_{x',A}$.
    *   The posterior covariance is **independent of the observations** $y_i$ and only decreases as more data is added.

* **Sampling**
    *   **Cholesky Method:** Sample a discretized subset $f$ via $f = K^{1/2}\epsilon + \mu$, where $\epsilon \sim N(0, I)$.
    *   **Forward Sampling:** Obtain samples one-by-one using the product rule $p(f_1, \dots, f_n) = \prod_{i=1}^n p(f_i | f_{1:i-1})$.
    *   **Complexity:** Both exact methods require $O(n^3)$ time due to matrix inversion.

* **Kernel Functions**
    *   **Purpose:** Kernels encode the **"shape"** and **smoothness** of functions.
    *   **Common Kernels:**
        *   **Linear:** $k(x, x') = x^\top x'$.
        *   **Gaussian (RBF):** $k(x, x'; h) = \exp\left( -\frac{\|x-x'\|^2}{2h^2} \right)$; defines an **infinitely dimensional** feature space.
        *   **Laplace:** $k(x, x'; h) = \exp\left( -\frac{\|x-x'\|}{h} \right)$.
    *   **Matérn:** Generalization that controls differentiability via parameter $\nu$.
    *   **Composition Rules:** New kernels can be created by **summing**, **multiplying**, or **scaling** existing kernels.

* **Model Selection**
    *   **Empirical Bayes:** Instead of cross-validation, maximize the **marginal likelihood** of the data.
    *   **Objective (MLL Loss):** $\text{arg min}_\theta \frac{1}{2} y^\top K^{-1}_{y,\theta} y + \frac{1}{2} \log \det(K_{y,\theta})$.
        *   Term 1 is **"goodness of fit"**; Term 2 is model **"volume/complexity"**.
    *   **Hyperprior:** A prior $p(\theta)$ on model parameters can be used to obtain a **MAP estimate**.

* **Approximations**
    *   **Local Methods:** Cut off the "tails" of the kernel to only condition on "close" samples.
    *   **Random Fourier Features (RFF):** Approximate the kernel with a low-dimensional feature map $\phi: \mathbb{R}^d \to \mathbb{R}^m$.
    *   **Inducing Points:** Summarize data around a set of points $U = \{x_1, \dots, x_k\}$.
        *   **SoR (Subset of Regressors):** Forgets all variance/covariance in the conditional.
        *   **FITC (Fully Independent Training Conditional):** Keeps variances but forgets covariances.

### **Variational Inference**
* **Laplace Approximation**
    *   **Concept:** A simple Gaussian approximation obtained from a **second-order Taylor expansion** of the log-posterior around its mode (the MAP estimate).
    *   **Posterior Mode:** $\hat{\theta} = \arg \max_\theta p(\theta | y)$.
    *   **Precision Matrix ($\Lambda$):** The negative Hessian of the log-posterior evaluated at the mode: $\Lambda = -H_\theta \log p(\theta | y) |_{\theta=\hat{\theta}}$.
    *   **Resulting Approximation:** $q(\theta) = N(\theta; \hat{\theta}, \Lambda^{-1})$.
    *   **Limitations:** It can be extremely overconfident if the true posterior is not approximately Gaussian.

* **Predictions with a Variational Posterior**
    *   **General Prediction:** $p(y^* | x^*, D) \approx \int p(y^* | x^*, \theta) q_\lambda(\theta) d\theta$.
    *   **Monte Carlo Approximation:** $E_{\theta \sim q_\lambda} [p(y^* | x^*, \theta)] \approx \frac{1}{m} \sum_{j=1}^m p(y^* | x^*, \theta^{(j)})$ where $\theta^{(j)} \sim q_\lambda$.
    *   **Function-space View (Logistic Regression):** Replaces high-dimensional weight-space integrals with a one-dimensional integral over the latent value $f^* = w^\top x^*$: $p(y^* | x^*) \approx \int \sigma(y^* f^*) \cdot N(f^*; \hat{w}^\top x^*, x^{*\top} \Lambda^{-1} x^*) df^*$.

* **Blueprint of Variational Inference**
    *   **Variational Family ($Q$):** A class of tractable distributions (e.g., independent/mean-field Gaussians) characterized by parameters $\lambda$.
    *   **Optimization Goal:** Find $q \in Q$ that minimizes the **Kullback-Leibler (KL) divergence** to the true posterior: $q^* = \arg \min_{q \in Q} KL(q || p)$.

* **Information Theory**
    *   **Surprise:** $S[u] = -\log u$.
    *   **Entropy:** The average surprise of a distribution: $H[p] = E_{x \sim p}[-\log p(x)]$.
    *   **Gaussian Entropy:** $H[N(\mu, \Sigma)] = \frac{1}{2} \log \det(2\pi e \Sigma)$.
    *   **Cross-Entropy:** $H[p || q] = E_{x \sim p}[-\log q(x)] = H[p] + KL(p || q)$.
    *   **KL Divergence:** $KL(q || p) = \int q(\theta) \log \frac{q(\theta)}{p(\theta)} d\theta$.
    *   **Forward vs. Reverse KL:**
        *   **Reverse (Exclusive):** $\arg \min_q KL(q || p)$; tends to be **mode-seeking** and underestimates variance.
        *   **Forward (Inclusive):** $\arg \min_q KL(p || q)$; results in **moment matching** (e.g., a Gaussian $q$ will match the first two moments of $p$).

* **Evidence Lower Bound (ELBO)**
    *   **Derivation:** Because $\log p(y) = KL(q || p(\cdot | y)) + L(q)$, and $KL \ge 0$, then $\log p(y) \ge L(q)$.
    *   **ELBO Formulas:**
        *   $L(q) = E_{\theta \sim q}[\log p(y, \theta)] + H[q]$.
        *   $L(q) = E_{\theta \sim q}[\log p(y | \theta)] - KL(q || p_{prior})$.
    *   **Reparameterization Trick:** For a random variable $\theta = g(\epsilon; \lambda)$ where $\epsilon$ is sampled from a base distribution $\phi$ independent of $\lambda$:
        *   $\nabla_\lambda E_{\theta \sim q_\lambda} [f(\theta)] = E_{\epsilon \sim \phi} [\nabla_\lambda f(g(\epsilon; \lambda))]$.
        *   For Gaussians: $\theta = \mu + \Sigma^{1/2}\epsilon$, where $\epsilon \sim N(0, I)$.
    *   **Curiosity and Conformity:** The ELBO trades off **Conformity** (minimizing "Energy" or negative log-likelihood to fit data) and **Curiosity** (maximizing entropy to maintain uncertainty).

### **Markov Chain Monte Carlo Methods**
* **Markov Chains**
    *   **Goal:** Estimate expectations $E_{\theta \sim p}[f(\theta)]$ using samples $\theta^{(i)}$ when the **normalizing constant $Z$** is unknown in $p(x) = \frac{1}{Z}q(x)$,.
    *   **Definition:** A stochastic process $(X_t)_{t\in\mathbb{N}_0}$ where the **Markov property** holds: $X_{t+1} \perp X_{0:t-1} | X_t$.
    *   **Transition Matrix ($P$):** Defined by $p(x' | x) = P(X_{t+1} = x' | X_t = x)$.
    *   **State Update:** The distribution at time $t+1$ is $q_{t+1} = q_t P$.
    *   **Stationarity ($\pi$):** A distribution $\pi$ is stationary if $\pi = \pi P$.
    *   **Ergodicity:** A chain is ergodic if it is **irreducible** (all states reachable) and **aperiodic**,. It converges to a unique stationary distribution $\pi$ regardless of the initial $q_0$.
    *   **Detailed Balance Equation:** A sufficient condition for stationarity: $\pi(x)p(x' | x) = \pi(x')p(x | x')$,.
    *   **Ergodic Theorem:** Allows estimating expectations using a single chain: $\frac{1}{n} \sum_{i=1}^n f(x_i) \xrightarrow{a.s.} E_{x \sim \pi}[f(x)]$.
    *   **Burn-in:** Discarding initial $t_0$ samples before the chain reaches the stationary distribution.

* **Elementary Sampling Methods**
    *   **Metropolis-Hastings (MH):**
        *   Uses a proposal distribution $r(x' | x)$ and an **acceptance probability**:
        *   $\alpha(x' | x) = \min \left\{ 1, \frac{q(x')r(x | x')}{q(x)r(x' | x)} \right\}$.
        *   Corrects for bias in proposals so the chain satisfies detailed balance.
    *   **Gibbs Sampling:**
        *   Special case of MH where you update one coordinate at a time: $x_i \sim p(x_i | x_{-i})$.
        *   The **acceptance probability $\alpha$ is always 1**.

* **Sampling as Optimization**
    *   **Gibbs/Boltzmann Distributions:** $p(x) = \frac{1}{Z} \exp(-f(x))$, where $f(x)$ is the **energy function** (often the negative log-posterior).
    *   **Langevin Dynamics:**
        *   Uses gradients to bias proposals toward low-energy (high-probability) regions.
        *   **Unadjusted Langevin Algorithm (ULA):** $\theta_{k+1} = \theta_k - \nabla f(\theta_k)\Delta t_k + \sqrt{2\Delta t_k} \epsilon$, where $\epsilon \sim N(0, I)$.
        *   Decomposition: **Drift** (conformity to data) + **Noise** (curiosity/exploration).
    *   **Stochastic Gradient Langevin Dynamics (SGLD):** Replaces exact gradients with mini-batch estimates, transitioning from stochastic gradient ascent to Langevin dynamics as the learning rate $\eta_t$ decreases.
    *   **Hamiltonian Monte Carlo (HMC):**
        *   Introduces auxiliary **momentum variables** $y \sim N(0, mI)$ to help the chain "jump" between modes.
        *   **Hamiltonian ($H$):** $H(x, y) = \frac{1}{2m} \|y\|_2^2 + f(x)$.
        *   Simulated via the **Leapfrog method** (discretization) followed by an MH acceptance step to correct for numerical errors.
### **Deep Learning**
* **Artificial Neural Networks (ANNs)**
    *   **Definition:** Nested linear functions composed with nonlinearities $\phi$.
        *   **Forward Pass:** $f(x; \theta) = \phi(W_L\phi(W_{L-1}(\dots\phi(W_1x))))$ where $\theta = [W_1, \dots, W_L]$.
    *   **Activation Functions:**
        *   **Tanh:** $\text{Tanh}(z) = \frac{\exp(z)-\exp(-z)}{\exp(z)+\exp(-z)} \in (-1, 1)$.
        *   **ReLU:** $\text{ReLU}(z) = \max\{z, 0\}$.
    *   **Classification:**
        *   **Softmax:** $\sigma_i(f) = \frac{\exp(f_i)}{\sum_{j=1}^c \exp(f_j)}$ (generalizes the logistic function).
    *   **Maximum Likelihood (MLE):** Equivalent to minimizing **Cross-Entropy Loss**:
        *   $\ell_{ce}(\theta; D) \approx -\frac{1}{n} \sum_{i=1}^n \log q_\theta(y_i | x_i)$.
    *   **Backpropagation:** Repeated application of the chain rule to compute gradients efficiently.

* **Bayesian Neural Networks (BNNs)**
    *   **Generative Model:**
        *   **Prior:** $\theta \sim N(0, \sigma^2_p I)$.
        *   **Likelihood:** $y | x, \theta \sim N(f(x; \theta), \sigma^2_n)$.
    *   **MAP Estimation:** Equivalent to **Squared Error + $\ell_2$-regularization (Weight Decay)**:
        *   $\hat{\theta}_{MAP} = \arg \min_\theta \left[ \frac{1}{2\sigma^2_p} \|\theta\|_2^2 + \frac{1}{2\sigma^2_n} \sum_{i=1}^n (y_i - f(x_i; \theta))^2 \right]$.
    *   **Heteroscedastic Noise:** Modeling input-dependent variance $\sigma^2(x; \theta) = \exp(f_2(x; \theta))$.
    *   **Loss:** $\frac{1}{2} \sum \left[ \log \sigma^2(x_i; \theta) + \frac{(y_i - \mu(x_i; \theta))^2}{\sigma^2(x_i; \theta)} \right]$.

* **Approximate Probabilistic Inference**
    *   **Variational Inference (Bayes by Backprop):**
        *   Uses **Reparameterization Trick**: $\theta = \mu + \Sigma^{1/2}\epsilon$ with $\epsilon \sim N(0, I)$.
        *   Optimizes **ELBO**: $L(q) = E_{\theta \sim q}[\log p(y_{1:n} | x_{1:n}, \theta)] - KL(q || p_{prior})$.
    *   **Predictive Posterior:** Averaging predictions over $m$ sampled models:
        *   $p(y^* | x^*, D) \approx \frac{1}{m} \sum_{j=1}^m p(y^* | x^*, \theta^{(j)})$.
    *   **Variance Decomposition:**
        *   $\text{Var}[y^* | x^*] \approx \frac{1}{m} \sum \sigma^2(x^*; \theta^{(j)}) + \frac{1}{m-1} \sum (\mu(x^*; \theta^{(j)}) - \bar{\mu}(x^*))^2$ (Aleatoric + Epistemic).
    *   **MCMC & SWAG:**
        *   **SGLD:** Injects noise into SGD updates to converge to the posterior.
        *   **SWAG:** Collects running averages of weights $\mu$ and second moments $A$ to form a Gaussian posterior.
    *   **Dropout/Dropconnect:**
        *   Interpreted as VI with a spike-and-slab posterior: $q_j(\theta_j | \lambda_j) = p\delta_0(\theta_j) + (1-p)\delta_{\lambda_j}(\theta_j)$.
    *   **Ensembles:** Training $m$ models on bootstrapped random subsamples of data.
    *   **Stein Variational Gradient Descent (SVGD):**
        *   Diverse ensembles using a **repulsion term** to avoid particle collapse:
        *   $\phi_{repulsion} = \sum_j \frac{1}{h^2}(\theta - \theta^{(j)})k(\theta, \theta^{(j)})$.

* **Calibration**
    *   **Definition:** Accuracy should match confidence.
    *   **Reliability Diagrams:** Binned visualization of accuracy vs. confidence.
    *   **Calibration Error:** **ECE** (Expected Calibration Error).
    *   **Temperature Scaling:** Simple post-hoc heuristic for probability smoothing:
        *   $q_i = \sigma\left(\frac{z_i}{T}\right)$.

### **Active Learning**
* **Conditional Entropy**
    *   **Definition:** Average surprise about random vector $X$ given $Y$: $H[X | Y] = \mathbb{E}_{y \sim p(y)}[H[X | Y = y]]$.
    *   **Chain Rule for Entropy:** $H[X, Y] = H[Y] + H[X | Y] = H[X] + H[Y | X]$.
    *   **Bayes’ Rule for Entropy:** $H[X | Y] = H[Y | X] + H[X] - H[Y]$.
    *   **Monotonicity:** $H[X | Y] \le H[X]$ ("Information never hurts").

* **Mutual Information (MI)**
    *   **Definition (Information Gain):** Reduction in uncertainty about $X$ after observing $Y$: $I(X; Y) = H[X] - H[X | Y]$.
    *   **Symmetry:** $I(X; Y) = I(Y; X)$.
    *   **MI of Gaussians:** For $X \sim N(\mu, \Sigma)$ and noisy observation $Y = X + \epsilon$ with $\epsilon \sim N(0, \sigma^2_n I)$:
        *   $I(X; Y) = \frac{1}{2} \log \det(I + \sigma^{-2}_n \Sigma)$.
    *   **Interaction Information:** $I(X; Y; Z) = I(X; Y) - I(X; Y | Z)$.
    *   **Redundancy:** Positive interaction (conditioning on $Z$ decreases MI).
    *   **Synergy:** Negative interaction (learning $Z$ increases what $Y$ tells us about $X$).
    *   **Utility Objective:** Find subset $S \subseteq \mathcal{X}$ of size $n$ maximizing $I(S) = I(f_S; y_S) = H[f_S] - H[f_S | y_S]$.

* **Submodularity of Mutual Information**
    *   **Marginal Gain:** $\Delta F(x | A) = F(A \cup \{x\}) - F(A)$.
    *   **Submodularity Definition:** For $A \subseteq B \subseteq \mathcal{X}$, $\Delta F(x | A) \ge \Delta F(x | B)$.
        *   Equivalent to "diminishing returns" or the absence of synergy.
    *   **Monotonicity:** $F(A) \le F(B)$.
    *   **Theorem:** The MI objective $I(S)$ is **monotone submodular**.

* **Maximizing Mutual Information**
    *   **Complexity:** Maximizing MI is NP-hard.
    *   **Greedy Approximation:** Picking points $x_1 \dots x_n$ one-by-one provides a $(1 - 1/e)$-approximation.
    *   **Uncertainty Sampling (Regression):** Greedily pick $x_{t+1} = \arg \max_{x \in \mathcal{X}} \sigma_t^2(x)$.
    *   **Heteroscedastic Noise:** Pick $x_{t+1} = \arg \max_{x \in \mathcal{X}} \frac{\sigma_t^2(x)}{\sigma_n^2(x)}$.
    *   **Classification (BALD):** Bayesian Active Learning by Disagreement.
        *   $\arg \max_{x \in \mathcal{X}} [H[y_x | D] - \mathbb{E}_{\theta | D}[H[y_x | \theta]]]$.
        *   Subtracts aleatoric uncertainty from total uncertainty to find where models disagree most.

* **Learning Locally: Transductive Active Learning**
    *   **Goal:** Select observations to best predict a specific target value $f(x^*)$ rather than the whole function.
    *   **Transductive Objective:** $x_{t+1} = \arg \max_{x \in \mathcal{X}'} I(f_{x^*}; y_x | D_t) = \arg \min_{x \in \mathcal{X}'} H[f_{x^*} | D_t, y_x]$.
    *   **Tradeoff:** Balances **diversity** (global info) with **relevance** (specific to $x^*$).

### **Bayesian Optimization**
*   **Setup and Objective:** 
    *   Find the maximum of an unknown "black-box" function $f^*$ such that $x^* = \arg \max_{x \in \mathcal{X}} f^*(x)$.
    *   **Observations:** Evaluations are noisy and costly: $y_t = f^*(x_t) + \epsilon_t$.
    *   **Prior:** Typically assumes a **Gaussian process (GP)** prior to encode function smoothness.

* **Exploration-Exploitation Dilemma**
    *   **Exploration:** Selecting points with **high posterior variance** to gain information about the model.
    *   **Exploitation:** Selecting points with **high posterior mean** and low variance where the function is expected to be high.
    *   **Principle:** Also known as the principle of **curiosity and conformity**.

* **Online Learning and Bandits**
    *   **Goal:** Maximize cumulative reward $\sum_{t=1}^T f^*(x_t)$.
    *   **Regret ($R_T$):** The additive loss compared to the static optimum: $R_T = \sum_{t=1}^T (\max_x f^*(x) - f^*(x_t))$.
    *   **Sublinear Regret:** Required for convergence to the true optimum: $\lim_{T \to \infty} \frac{R_T}{T} = 0$.

* **Acquisition Functions**
    *   **Strategy:** Use a utility function $\mathcal{F}$ to greedily pick the next point: $x_t = \arg \max_{x \in \mathcal{X}} \mathcal{F}(x; \mu_{t-1}, k_{t-1})$.

    * **Upper Confidence Bound (UCB)**
        *   **Optimism in the Face of Uncertainty:** $x_{t+1} = \arg \max_{x \in \mathcal{X}} \mu_t(x) + \beta_{t+1} \sigma_t(x)$.
        *   **Confidence Parameter ($\beta_t$):**
            *   **Bayesian Setting:** $\beta_t(\delta) = \mathcal{O}(\sqrt{\log(|\mathcal{X}|t/\delta)})$.
            *   **Frequentist Setting:** $\beta_t(\delta) = \|f^*\|_k + \sigma_n \sqrt{2(\gamma_t + \log(1/\delta))}$.
        *   **Regret Bound:** $R_T = \mathcal{O}(\beta_T(\delta) \sqrt{\gamma_T T})$ where $\gamma_T$ is the **maximum information gain**.

    * **Improvement-based Methods**
        *   **Improvement ($I_t(x)$):** Measured against the current running optimum $\hat{f}_t$: $I_t(x) = (f(x) - \hat{f}_t)_+$.
        *   **Probability of Improvement (PI):** $x_{t+1} = \arg \max_{x \in \mathcal{X}} \Phi \left( \frac{\mu_t(x) - \hat{f}_t}{\sigma_t(x)} \right)$.
        *   **Expected Improvement (EI):** $x_{t+1} = \arg \max_{x \in \mathcal{X}} \mathbb{E}[I_t(x) | x_{1:t}, y_{1:t}]$.

    * **Thompson Sampling**
        *   **Probability Matching:** Select points according to the probability that they are optimal.
        *   **Implementation:** Sample a function $\tilde{f}_{t+1}$ from the posterior and maximize it: $x_{t+1} = \arg \max_{x \in \mathcal{X}} \tilde{f}_{t+1}(x)$.

    * **Information-Directed Sampling (IDS)**
        *   **Information Ratio ($\Psi_t$):** Minimizes $\Psi_t(x) = \frac{\Delta(x)^2}{I_t(x)}$.
        *   **Surrogate Regret:** Uses confidence bounds: $\hat{\Delta}_t(x) = \max_{x' \in \mathcal{X}} u_t(x') - l_t(x)$.

    * **LITE: Probability of Maximality**
        *   **Objective:** Estimate $\pi(x) = P(f(x) = \max_{x'} f(x'))$.
        *   **LITE Approximation:** $\pi(x) \approx \Phi \left( \frac{\mu_t(x) - \kappa^*}{\sigma_t(x)} \right)$ with $\kappa^*$ chosen such that the distribution integrates to one.
        *   **Dichotomy:** Exploration is achieved through **optimism** (variance bonus) and **decision uncertainty** (entropy regularization).

### **Markov Decision Processes (MDPs)**
* **Setup and Definitions**
    *   **Definition (MDP):** A finite Markov decision process is specified by states $\mathcal{X}$, actions $\mathcal{A}$, **transition probabilities** (dynamics model) $p(x' | x, a)$, and a **reward function** $r(x, a)$.
    *   **Policy:** A function mapping states to a distribution over actions: $\pi(a | x) = P(A_t = a | X_t = x)$.
    *   **Discounted Payoff ($G_t$):** The total reward from time $t$ with discount factor $\gamma \in [0, 1)$: $G_t = \sum_{m=0}^\infty \gamma^m R_{t+m}$.
    *   **State Value Function ($v^\pi$):** Expected payoff starting from state $x$: $v^\pi(x) = E_\pi[G_t | X_t = x]$.
    *   **Action-Value Function ($q^\pi$):** Expected payoff starting from $x$, playing $a$, then following $\pi$: $q^\pi(x, a) = r(x, a) + \gamma \sum_{x' \in \mathcal{X}} p(x' | x, a) v^\pi(x')$.

* **Bellman Expectation Equation**
    *   **Recursive Form:** $v^\pi(x) = r(x, \pi(x)) + \gamma \sum_{x' \in \mathcal{X}} p(x' | x, \pi(x)) v^\pi(x')$.
    *   **Stochastic Policies:** $v^\pi(x) = \sum_{a \in \mathcal{A}} \pi(a | x) q^\pi(x, a)$.

* **Policy Evaluation**
    *   **Exact Solution:** For a fixed policy, the value function is found by solving the linear system $\mathbf{v}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi$.
    *   **Fixed-point Iteration:** $v$ is the unique fixed-point of the contraction mapping $\mathcal{B}^\pi v = r^\pi + \gamma P^\pi v$.
    *   **Convergence:** The estimate $v^\pi_t$ converges to $v^\pi$ at an exponential rate: $\|v^\pi_t - v^\pi\|_\infty \le \gamma^t \|v^\pi_0 - v^\pi\|_\infty$.

* **Policy Optimization**
    *   **Optimal Value Function ($v^*$):** $v^*(x) = \max_\pi v^\pi(x)$.
    *   **Bellman Optimality Equation:** 
        *   $v^*(x) = \max_{a \in \mathcal{A}} [r(x, a) + \gamma \sum_{x' \in \mathcal{X}} p(x' | x, a) v^*(x')]$.
        *   $q^*(x, a) = r(x, a) + \gamma \sum_{x' \in \mathcal{X}} p(x' | x, a) \max_{a' \in \mathcal{A}} q^*(x', a')$.
    *   **Policy Iteration:** Iterates between **computing $v^\pi$** (evaluation) and **updating $\pi$** to be greedy with respect to $v^\pi$ (improvement).
    *   **Value Iteration:** Iteratively applies the Bellman optimality update: $v(x) \leftarrow \max_{a \in \mathcal{A}} q(x, a)$.

* **Partial Observability (POMDP)**
    *   **Definition:** Extension of MDP where the agent receives **noisy observations** $Y_t$ via probabilities $o(y | x)$ rather than the state $X_t$ directly.
    *   **Belief State ($b_t$):** A distribution over possible states: $b_t(x) = P(X_t = x | y_{1:t}, a_{1:t-1})$.
    *   **Belief Update:** Filtering step to incorporate new data: $b_{t+1}(x) = \frac{1}{Z} o(y_{t+1} | x) \sum_{x' \in \mathcal{X}} p(x | x', a_t) b_t(x')$.
    *   **Belief-state MDP:** A POMDP can be solved as an MDP where the states are beliefs $b \in \Delta_\mathcal{X}$.

### Tabular Reinforcement Learning
* **The Reinforcement Learning Problem**
    *   **Goal:** Probabilistic planning in **unknown environments** where dynamics $p$ and rewards $r$ are not given a-priori.
    *   **Trajectories ($\tau$):** A sequence of transitions $\tau_i = (x_i, a_i, r_i, x_{i+1})$ where $x_i$ is the state, $a_i$ the action, and $r_i$ the reward.
    *   **Markovian Structure:** Outcomes $(x_{t+1}, r_t)$ are conditionally independent of the past given the current state-action pair $(x_t, a_t)$.
    *   **Settings:** 
        *   **Episodic:** The agent is reset to an initial state after each round (episode).
    *   **Continuous/Online:** Learning happens in real-time without resets.

* **Model-based Approaches**
    *   **Concept:** Learn the underlying MDP (dynamics $p$ and rewards $r$) and then plan a policy.
    *   **Maximum Likelihood Estimates (MLE):**
        *   **Dynamics:** $\hat{p}(x' | x, a) = \frac{N(x' | x, a)}{N(a | x)}$, where $N$ denotes transition counts.
        *   **Rewards:** $\hat{r}(x, a) = \frac{1}{N(a | x)} \sum_{t: x_t=x, a_t=a} r_t$.

* **Balancing Exploration and Exploitation**
    *   **$\epsilon$-greedy:** Pick a random action with probability $\epsilon_t$, otherwise pick the best action under the current model.
    *   **GLIE (Greedy in the Limit with Infinite Exploration):** A condition for convergence requiring all pairs to be visited infinitely often and the policy to converge to a greedy one.
    *   **Softmax/Boltzmann Exploration:** $\pi_\lambda(a | x) \propto \exp \left( \frac{1}{\lambda} Q^*(x, a) \right)$.
    *   **Optimism (Rmax Algorithm):** 
    *   Assumes unknown transitions lead to a "fairy-tale" state $x^*$ with maximum rewards $R_{max}$.
    *   Convergence to an $\epsilon$-optimal policy requires a number of samples $N(a | x) \ge \frac{R_{max}^2}{2\epsilon^2} \log \frac{2}{\delta}$.

* **Model-free Approaches**
    *   **Concept:** Estimate the value function directly without learning the environment model $p$ or $r$.
    *   **Temporal-Difference (TD) Learning:** **On-policy** value estimation.
        *   **Update:** $V^\pi(x) \leftarrow (1 - \alpha_t)V^\pi(x) + \alpha_t(r + \gamma V^\pi(x'))$.
    *   **SARSA:** **On-policy** control (State-Action-Reward-State-Action).
        *   **Update:** $Q^\pi(x, a) \leftarrow (1 - \alpha_t)Q^\pi(x, a) + \alpha_t(r + \gamma Q^\pi(x', a'))$.
    *   **Q-learning:** **Off-policy** control; estimates the optimal Q-function $q^*$ directly.
        *   **Update:** $Q^*(x, a) \leftarrow (1 - \alpha_t)Q^*(x, a) + \alpha_t(r + \gamma \max_{a' \in \mathcal{A}} Q^*(x', a'))$.
    *   **Optimistic Q-learning:**
        *   **Initialization:** $Q^*(x, a) = V_{max} \prod_{t=1}^{T_{init}} (1 - \alpha_t)^{-1}$, where $V_{max} = \frac{R_{max}}{1 - \gamma}$.
        *   Allows for fast convergence in sparse MDPs with $O(Tm)$ time complexity.

### Model Free Reinforcement Learning
* **Tabular RL as Optimization**
    *   **Concept:** Reinterprets Temporal-Difference (TD) learning as **stochastic semi-gradient descent**.
    *   **Loss Function:** $l(\theta; x, r, x') = \frac{1}{2}(r + \gamma \theta_{old}(x') - \theta(x))^2$.
    *   **TD Error ($\delta_{TD}$):** $\theta(x) - (r + \gamma \theta_{old}(x'))$.
    *   **Semi-gradient:** Called "semi" because it performs gradient descent with respect to **bootstrapping estimates** (which are treated as constants) rather than the true value function.

* **Value Function Approximation**
    *   **Scaling:** Approximating values with parameters $\theta$ (e.g., $Q(x, a; \theta) = \theta^\top \phi(x, a)$) to exploit **smoothness** across large state-action spaces.
    *   **Deep Q-Networks (DQN):**
        *   **Experience Replay:** Stores transitions in a buffer $D$.
        *   **Target Networks:** Uses a fixed network $\theta_{old}$ to stabilize moving optimization targets.
    *   **Double DQN (DDQN):** Addresses **maximization bias** (overestimation of noisy Q-values) by using the current network to select actions and the target network to evaluate them.

* **Policy Approximation**
    *   **Definition:** Directly parameterizes the policy $\pi(a | x; \phi)$.
    *   **Policy Value Function ($j(\phi)$):** $E_\pi [\sum_{t=0}^\infty \gamma^t R_t]$.
    *   **Score Gradient Estimator (REINFORCE):** $\nabla_\phi j(\phi) \approx E_{\tau \sim \Pi_\phi} [G_0 \nabla_\phi \log \Pi_\phi(\tau)]$.
    *   **Baselines:** Subtracting a term $b$ from the return $G_0$ can **dramatically reduce variance** without introducing bias.
    *   **Downstream Returns:** $G_{t:T} = \sum_{m=0}^{T-1-t} \gamma^m R_{t+m}$ (also called "reward to go").

* **On-policy Actor-Critics**
    *   **Architecture:** Combines an **Actor** (parameterized policy) and a **Critic** (value function approximation).
    *   **Advantage Function:** $A^\pi(x, a) = Q^\pi(x, a) - V^\pi(x)$, which measures if an action is better than the average policy behavior.
    *   **Policy Gradient Theorem:** $\nabla_\phi j(\phi) \propto E_{x \sim \rho^\infty_\phi} E_{a \sim \pi_\phi(\cdot|x)} [q^\pi(x, a) \nabla_\phi \log \pi_\phi(a | x)]$.
    *   **Trust Regions (TRPO/PPO):** Uses a **KL-divergence constraint** to ensure the policy doesn't change too much in a single step, which allows for safer sample reuse.

* **Off-policy Actor-Critics**
    *   **DDPG:** Extends DQN to continuous action spaces by using a deterministic policy $\pi_\phi(x)$ to approximate the `argmax` over actions.
    *   **Reparameterization Trick (SVG):** Allows for gradients through randomized policies if the action can be written as $a = g(\epsilon; x, \phi)$ for independent noise $\epsilon$.

* **Maximum Entropy Reinforcement Learning (MERL)**
    *   **Objective:** Maximize reward **plus entropy**: $j_\lambda(\phi) = j(\phi) + \lambda H[\Pi_\phi]$.
    *   **Control as Inference:** Frames RL as minimizing the KL-divergence between the agent's trajectory distribution and a distribution weighted by rewards.
    *   **Soft Actor-Critic (SAC):** An off-policy algorithm that uses **soft Q-functions** and reparameterization to encourage exploration and robustness.

* **Learning from Preferences**
    *   **Context:** Used when numerical rewards are hard to define, such as in **alignment for Large Language Models (LLMs)**.
    *   **Bradley-Terry Model:** Models preference probability as $\sigma(r(y_A | x) - r(y_B | x))$.
    *   **RLHF:** Two stages: 1) Learn a reward model $r_\theta$ from human rankings, 2) Optimize the policy using PPO.
    *   **Direct Preference Optimization (DPO):** Skips the reward model stage by directly optimizing the policy to prefer favored responses based on their relative likelihood under a reference model.

### Model-Based Reinforcement Learning
* **Introduction: The World Model**
    *   **Concept:** Instead of learning a value function directly, the agent learns an approximate **dynamics model** $f \approx p$ and **reward model** $r$, collectively known as a **world model**.
    *   **Benefit:** Leveraging a world model allows the agent to anticipate consequences many steps ahead, making it significantly more **sample-efficient** than model-free methods.
    *   **Feedback Loops:** Unlike supervised learning, the data in the replay buffer depends on the agent's policy, creating feedback loops between the model and action selection.

* **Planning**
    *   **13.1.1 Deterministic Dynamics:** $x_{t+1} = f(x_t, a_t)$.
        *   **Model Predictive Control (MPC):** The agent plans over a finite horizon $H$, executes the first action, and then replans at the next step.
        *   **Objective:** Maximize $J_H(a_{t:t+H-1}) = \sum_{\tau=t}^{t+H-1} \gamma^{\tau-t} r(x_\tau, a_\tau)$.
        *   **Sparse Rewards:** To handle horizons where rewards aren't immediately reachable, a **long-term value estimate** $V(x_{t+H})$ is added to the tail of the sum.
    *   **Stochastic Dynamics:** Uses **trajectory sampling** (averaging over multiple potential futures).
        *   **Reparameterization Trick:** If $x_{t+1} = g(\epsilon; x_t, a_t)$, analytic gradients can be computed for the objective by backpropagating through the stochastic transitions.
    *   **Parametric Policies:** Decisions can be "stored" in a policy $\pi_\phi(x_t)$ to replace expensive online planning with cheap offline evaluation.

* **Learning the Model**
    *   **Supervised Perspective:** Dynamics and rewards are learned off-policy from a replay buffer $D$ using standard regression techniques.
    *   **The Pitfall of Point Estimates:** Planning often **overfits** to small errors in a point-estimate model (MAP), and these errors compound over long horizons.
    *   **Probabilistic Inference:** Capturing **epistemic uncertainty** (uncertainty about the model) helps the agent be robust to model inaccuracies.
    *   **Key Algorithms:**
        *   **PILCO:** Uses Gaussian Processes and moment matching for high sample efficiency.
    *   **PETS:** Uses ensembles of neural networks and trajectory sampling to account for both aleatoric and epistemic uncertainty.

* **Exploration and Safety**
    *   **Thompson Sampling:** The agent samples one model $f$ from the posterior and optimizes its policy for that specific "hallucinated" world.
    *   **Optimistic Exploration:**
        *   **H-UCRL:** The agent optimizes for the best possible outcome among all **plausible models** $M(D)$.
        *   **Hallucinated Dynamics:** $f_{t,i}(x, a) = \mu_{t,i}(x, a) + \beta_{t,i} \eta_i(x, a) \sigma_{t,i}(x, a)$, where $\eta$ represents "luck" variables controlled by the agent.
    *   **Constrained/Safe Exploration:**
        *   **Pessimism for Safety:** The agent is **optimistic** about future rewards but **pessimistic** about constraint violations to ensure it avoids unsafe states $X_{unsafe}$.
        *   **Safety Filters:** A "backup" safe policy $\pi_{safe}$ is used to override potentially dangerous actions if they risk entering unrecoverable states.

