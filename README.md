# คู่มือสรุปสิ่งที่ควรรู้และสูตรคำนวณทุกสัปดาห์ (Week 1 - Week 14)
### Mathematics & Statistical Learning for Data Science (ชุดฝึกปฏิบัติการธุรกิจ)

เอกสารนี้รวบรวม **สิ่งที่ควรรู้, แนวคิดทางธุรกิจ, สูตรคณิตศาสตร์/สถิติรูปแบบปกติ (LaTeX)** และ **รูปแบบโค้ด Python (NumPy, SciPy, pandas, scikit-learn, statsmodels)** ครบถ้วนทั้ง 14 สัปดาห์ เพื่อใช้เป็นคู่มืออ้างอิงและทบทวนความเข้าใจอย่างเป็นระบบ

---

## สารบัญ (Table of Contents)
- [กฎทองและข้อตกลงร่วมของระบบฝึกปฏิบัติการ](#กฎทองและข้อตกลงร่วมของระบบฝึกปฏิบัติการ)
- [Week 1: เวกเตอร์ NumPy — Norm, Cosine Similarity, ระยะทาง Euclid](#week-1-เวกเตอร์-numpy--norm-cosine-similarity-ระยะทาง-euclid)
- [Week 2: เมทริกซ์ ระบบสมการ $Ax = b$ และ Determinant](#week-2-เมทริกซ์-ระบบสมการ-ax--b-และ-determinant)
- [Week 3: Rank, Null Space, Projection, Least Squares, Condition Number](#week-3-rank-null-space-projection-least-squares-condition-number)
- [Week 4: Eigenvalue, SVD, Low-Rank Approximation และ PCA](#week-4-eigenvalue-svd-low-rank-approximation-และ-pca)
- [Week 5: Python สำหรับ Statistical Learning (Linear vs KNN, Test MSE)](#week-5-python-สำหรับ-statistical-learning-linear-vs-knn-test-mse)
- [Week 6: Bias-Variance Trade-Off (พหุนามหลายดีกรี, Bias², Variance, KNN)](#week-6-bias-variance-trade-off-พหุนามหลายดีกรี-bias²-variance-knn)
- [Week 7: EDA, Bootstrap CI, Independent t-test และ Chi-square](#week-7-eda-bootstrap-ci-independent-t-test-และ-chi-square)
- [Week 8: Simple Linear Regression (SE, RSE, t-statistic, p-value, R²)](#week-8-simple-linear-regression-se-rse-t-statistic-p-value-r)
- [Week 9: Multiple Linear Regression (Adjusted R², AIC, BIC, CI vs PI)](#week-9-multiple-linear-regression-adjusted-r-aic-bic-ci-vs-pi)
- [Week 10: Dummy Variables, Interaction Terms, Diagnostics และ VIF](#week-10-dummy-variables-interaction-terms-diagnostics-และ-vif)
- [Week 11: Logistic Regression ธุรกิจ (Odds Ratio, Confusion Matrix, Threshold, Class Weight)](#week-11-logistic-regression-ธุรกิจ-odds-ratio-confusion-matrix-threshold-class-weight)
- [Week 12: LDA, QDA, Gaussian Naive Bayes และ KNN ธุรกิจ](#week-12-lda-qda-gaussian-naive-bayes-และ-knn-ธุรกิจ)
- [Week 13: GLM (Poisson Regression), ROC AUC, Class Imbalance](#week-13-glm-poisson-regression-roc-auc-class-imbalance)
- [Week 14: Cross-Validation (Validation Set, LOOCV, K-Fold, 1-SE Rule)](#week-14-cross-validation-validation-set-loocv-k-fold-1-se-rule)
- [ตารางสรุป Quick Reference ฟังก์ชัน Python ที่ใช้บ่อยที่สุด](#ตารางสรุป-quick-reference-ฟังก์ชัน-python-ที่ใช้บ่อยที่สุด)

---

## กฎทองและข้อตกลงร่วมของระบบฝึกปฏิบัติการ

1. **ดัชนี 0-Based Indexing:**
   - ดัชนีแถวและคอลัมน์ของข้อมูลนับเริ่มต้นจาก `0` เสมอ
   - "แถวที่ $a$ ถึง $b$" ในโจทย์ภาษาไทย เป็นการระบุช่วงปิด $[a, b]$ หมายถึงแถวลำดับที่ $a$ ถึง $b$ ครบถ้วน ใน Python ให้ใช้ `.iloc[a : b + 1]`
2. **การแบ่งชุดข้อมูลแบบคงที่ (Deterministic Splitting):**
   - การแบ่ง Train/Test หรือการแบ่ง Fold ใน Cross-Validation ใช้ตาม **ลำดับแถวเดิมของไฟล์เสมอ** โดย **ไม่สับเปลี่ยนข้อมูล** (`shuffle=False`, ห้ามใช้ `train_test_split` แบบสุ่ม หรือ `random_state`)
3. **องศาความเป็นอิสระ (Degrees of Freedom: `ddof`):**
   - ค่าเบี่ยงเบนมาตรฐานประชากร (Population SD / `StandardScaler` / `np.std(ddof=0)`): ใช้ $N$ ในตัวหาร (`ddof=0`)
   - ค่าเบี่ยงเบนมาตรฐานกลุ่มตัวอย่าง (Sample SD / `ddof=1`): ใช้ $n - 1$ ในตัวหาร (`ddof=1`) ใช้ในการคำนวณ t-test, Bootstrap, Standard Error, และกฎ 1-SE rule
4. **การป้องกันการรั่วไหลของข้อมูล (Data Leakage):**
   - เมื่อทำการ Standardize ข้อมูล (`StandardScaler()`) จะต้อง `fit` สถิติ $\mu, \sigma$ บน **Training Data หรือ Train Fold เท่านั้น** แล้วนำไป `transform` ข้อมูล Validation/Test
5. **การปัดเศษตัวเลข:**
   - ทุกคำตอบในระบบให้ปัดเศษตามทศนิยมที่กำหนด (โดยทั่วไปคือ 4 ตำแหน่ง เช่น `{ans:.4f}`)

---

## Week 1: เวกเตอร์ NumPy — Norm, Cosine Similarity, ระยะทาง Euclid

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- เวกเตอร์ (Vector) คือโครงสร้างพื้นฐานในการแทนคุณลักษณะ (Features) ของลูกค้าหรือสินค้า เช่น เวกเตอร์แสดงยอดใช้จ่ายในหมวดหมู่ต่างๆ
- **Norm (ความยาว):** บ่งบอกขนาดโดยรวมของกิจกรรม
- **ระยะทาง (Distance) vs ความคล้าย (Similarity):** 
  - ระยะทาง Euclidean วัดความห่างเชิงตำแหน่งกายภาพ (ได้รับผลกระทบจากขนาด/สเกลของเวกเตอร์)
  - Cosine Similarity วัดทิศทางของเวกเตอร์ (ไม่ขึ้นกับขนาดของเวกเตอร์ เหมาะกับการวัดความสนใจเชิงสัดส่วน เช่น สัดส่วนหมวดหมู่สินค้าที่ซื้อ)

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **$L_1$ Norm (Manhattan Norm):**
  $$\|x\|_1 = \sum_{i=1}^n |x_i|$$
- **$L_2$ Norm (Euclidean Norm):**
  $$\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x^\top x}$$
- **$L_\infty$ Norm (Max Norm):**
  $$\|x\|_\infty = \max_{1 \le i \le n} |x_i|$$
- **Euclidean Distance:**
  $$d(x, y) = \|x - y\|_2 = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$
- **Dot Product:**
  $$x \cdot y = x^\top y = \sum_{i=1}^n x_i y_i$$
- **Cosine Similarity & Cosine Distance:**
  $$\text{Cosine Similarity} = \cos(\theta) = \frac{x \cdot y}{\|x\|_2 \|y\|_2} = \frac{\sum x_i y_i}{\sqrt{\sum x_i^2} \sqrt{\sum y_i^2}}$$
  $$\text{Cosine Distance} = 1 - \cos(\theta)$$
- **Vector Projection (ภาพฉายของ $y$ ลงบน $x$):**
  $$\text{proj}_x(y) = \left( \frac{x \cdot y}{\|x\|_2^2} \right) x = \left( \frac{x^\top y}{x^\top x} \right) x$$

### 3. รูปแบบโค้ด Python
```python
import numpy as np

# Norms
norm_l1 = np.linalg.norm(x, ord=1)
norm_l2 = np.linalg.norm(x, ord=2)  # หรือ np.linalg.norm(x)
norm_inf = np.linalg.norm(x, ord=np.inf)

# Euclidean Distance & Dot Product
dist = np.linalg.norm(x - y)
dot_prod = np.dot(x, y)

# Cosine Similarity & Distance
cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
cos_dist = 1.0 - cos_sim

# Projection of y onto x
proj_y_on_x = (np.dot(x, y) / np.dot(x, x)) * x
```

---

## Week 2: เมทริกซ์ ระบบสมการ $Ax = b$ และ Determinant

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- เมทริกซ์แทนความสัมพันธ์เชิงเส้นระหว่างตัวแปร เช่น เมทริกซ์ปัจจัยการผลิต, เมทริกซ์ต้นทุน
- การแก้ระบบสมการ $Ax = b$ คือการหาจุดดุลยภาพหรือปริมาณปัจจัยที่สอดคล้องกับเป้าหมาย
- **Determinant:** บ่งบอกอัตราการขยายตัวของพื้นที่/ปริมาตรหลังการแปลงเชิงเส้น หาก $\det(A) = 0$ เมทริกซ์นั้นไม่มี Inverse (Singular Matrix) ระบบสมการไม่มีคำตอบเดี่ยว

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **Matrix Multiplication:**
  $$C = AB \quad \implies \quad c_{ij} = \sum_{k=1}^m a_{ik} b_{kj}$$
- **Matrix Inverse & Identity:**
  $$A A^{-1} = A^{-1} A = I_n$$
- **Linear System Solution:**
  $$Ax = b \quad \implies \quad x = A^{-1} b \quad (\text{เมื่อ } \det(A) \ne 0)$$
- **Determinant ($2 \times 2$ และ $3 \times 3$):**
  $$\det \begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$
- **Trace (ผลบวกบนเส้นทแยงมุมหลัก):**
  $$\text{tr}(A) = \sum_{i=1}^n a_{ii}$$
- **Frobenius Norm:**
  $$\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n a_{ij}^2} = \sqrt{\text{tr}(A^\top A)}$$

### 3. รูปแบบโค้ด Python
```python
import numpy as np

# Matrix Multiplication
C = A @ B  # หรือ np.matmul(A, B)

# Matrix Inverse & Determinant
A_inv = np.linalg.inv(A)
det_A = np.linalg.det(A)

# Solve Linear System Ax = b (เร็วกว่าและเสถียรกว่าการคูณ inv ตรงๆ)
x = np.linalg.solve(A, b)

# Trace & Frobenius Norm
tr_A = np.trace(A)
frob_norm = np.linalg.norm(A, 'fro')
```

---

## Week 3: Rank, Null Space, Projection, Least Squares, Condition Number

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- **Rank:** จำนวนมิติอิสระของข้อมูล หาก Rank ต่ำกว่าจำนวนคอลัมน์ แสดงว่ามีตัวแปรที่ซ้ำซ้อนกันเชิงเส้น (Multicollinearity)
- **Null Space:** ทิศทางของอินพุตที่ให้ผลลัพธ์เป็นศูนย์ ($Ax = 0$)
- **Least Squares:** เมื่อระบบสมการ $Ax = b$ ไม่มีคำตอบพอดี (Overdetermined system, $m > n$) เราหาคำตอบ $\hat{x}$ ที่ทำให้ระยะคลาดเคลื่อน $\|b - A\hat{x}\|_2^2$ ต่ำที่สุด
- **Condition Number ($\kappa$):** วัดความไวต่อความคลาดเคลื่อน (Sensitivity) หาก $\kappa$ มีค่าสูงมาก แสดงว่าเมทริกซ์เข้าใกล้ภาวะเอกฐาน (Ill-conditioned)

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **Rank-Nullity Theorem:**
  $$\text{rank}(A) + \dim(\text{Null}(A)) = n \quad (\text{เมื่อ } A \in \mathbb{R}^{m \times n})$$
- **Orthogonal Projection Matrix ลงบน Column Space ของ $A$:**
  $$P = A(A^\top A)^{-1} A^\top$$
- **Projection of $b$ onto $\text{Col}(A)$:**
  $$\hat{b} = P b = A(A^\top A)^{-1} A^\top b$$
- **Normal Equations (Ordinary Least Squares - OLS):**
  $$A^\top A \hat{x} = A^\top b \quad \implies \quad \hat{x} = (A^\top A)^{-1} A^\top b$$
- **Residual Vector:**
  $$e = b - \hat{b} = (I - P)b \quad \text{โดยที่ } e \perp \text{Col}(A) \implies A^\top e = 0$$
- **Condition Number:**
  $$\kappa(A) = \|A\|_2 \|A^{-1}\|_2 = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

### 3. รูปแบบโค้ด Python
```python
import numpy as np
import scipy.linalg

# Rank & Null Space
rank = np.linalg.matrix_rank(A)
null_basis = scipy.linalg.null_space(A)

# Projection Matrix & Projection Vector
P = A @ np.linalg.inv(A.T @ A) @ A.T
b_hat = P @ b

# Least Squares Solution
x_hat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
# หรือคำนวณตามสูตร OLS:
x_hat_formula = np.linalg.inv(A.T @ A) @ A.T @ b

# Residual Vector & Residual Norm
e = b - A @ x_hat
residual_norm = np.linalg.norm(e)

# Condition Number
cond_num = np.linalg.cond(A)
```

---

## Week 4: Eigenvalue, SVD, Low-Rank Approximation และ PCA

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- **Eigenvalues & Eigenvectors:** ทิศทางที่ไม่เปลี่ยนทิศหลังการแปลงเชิงเส้น ($Av = \lambda v$)
- **SVD (Singular Value Decomposition):** ใช้งานได้กับเมทริกซ์ทุกขนาด ($m \times n$) เป็นเครื่องมือแยกโครงสร้างสาระสำคัญออกจากสัญญาณรบกวน (Noise)
- **Low-Rank Approximation (Eckart-Young-Mirsky Theorem):** การประมาณเมทริกซ์เดิมด้วยเมทริกซ์ที่มี Rank ต่ำกว่า ($k$) ที่ดีที่สุดในแง่ Frobenius Norm
- **PCA (Principal Component Analysis):** การลดมิติข้อมูลโดยรักษาความแปรปรวน (Variance) สูงสุด

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **Eigenvalue Equation:**
  $$A v = \lambda v \iff (A - \lambda I)v = 0 \iff \det(A - \lambda I) = 0$$
- **Singular Value Decomposition (SVD):**
  $$A = U \Sigma V^\top = \sum_{i=1}^r \sigma_i u_i v_i^\top$$
  - $U$: Left singular vectors (ขนาด $m \times m$)
  - $\Sigma$: Diagonal matrix ของ singular values ($\sigma_1 \ge \sigma_2 \ge \dots \ge 0$)
  - $V$: Right singular vectors (ขนาด $n \times n$)
- **Rank-$k$ Approximation Matrix:**
  $$A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top = U_k \Sigma_k V_k^\top$$
- **Reconstruction Error:**
  $$\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$
- **PCA Explained Variance Ratio:**
  $$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^p \lambda_j} = \frac{\sigma_k^2}{\sum_{j=1}^p \sigma_j^2}$$

### 3. รูปแบบโค้ด Python
```python
import numpy as np
from sklearn.decomposition import PCA

# Eigen-decomposition
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Rank-k Approximation
k = 2
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
recon_error = np.linalg.norm(A - A_k, 'fro')

# PCA ด้วย scikit-learn (ข้อมูลต้องถูกศูนย์กลาง - Mean Center อัตโนมัติ)
pca = PCA(n_components=k)
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_
```

---

## Week 5: Python สำหรับ Statistical Learning (Linear vs KNN, Test MSE)

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- **Parametric Model (Linear Regression):** สมมติรูปแบบเชิงเส้น $\hat{y} = X\beta$ ตีความง่าย แต่มี Bias สูงถ้าความสัมพันธ์จริงไม่เป็นเส้นตรง
- **Non-Parametric Model (K-Nearest Neighbors - KNN):** ไม่สมมติรูปแบบสมการ ทำนายจากค่าเฉลี่ยของเพื่อนบ้าน $K$ ตัวที่ใกล้ที่สุด
- **Feature Scaling ใน KNN:** ระยะทางแบบ Euclidean อ่อนไหวต่อหน่วยวัด ต้องทำ Standardization ($\mu=0, \sigma=1$) ก่อนรัน KNN เสมอ

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
- **KNN Regressor:**
  $$\hat{f}(x_0) = \frac{1}{K} \sum_{x_i \in \mathcal{N}_K(x_0)} y_i$$
- **Standardization (Z-score):**
  $$z = \frac{x - \mu}{\sigma}, \quad \mu = \frac{1}{n} \sum x_i, \quad \sigma = \sqrt{\frac{1}{n} \sum (x_i - \mu)^2} \quad (ddof=0)$$

### 3. รูปแบบโค้ด Python
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# แบ่งข้อมูลแบบไม่สุ่ม (Deterministic Split)
train_df = df.iloc[:800]
test_df = df.iloc[800:]

# Linear Regression
lr = LinearRegression()
lr.fit(train_df[['x1', 'x2']], train_df['y'])
pred_lr = lr.predict(test_df[['x1', 'x2']])
mse_lr = mean_squared_error(test_df['y'], pred_lr)

# KNN Regression พร้อม Standardization
scaler = StandardScaler()
X_train_s = scaler.fit_transform(train_df[['x1', 'x2']])
X_test_s = scaler.transform(test_df[['x1', 'x2']])

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_s, train_df['y'])
pred_knn = knn.predict(X_test_s)
mse_knn = mean_squared_error(test_df['y'], pred_knn)
```

---

## Week 6: Bias-Variance Trade-Off (พหุนามหลายดีกรี, Bias², Variance, KNN)

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- **Expected Test MSE:** ประกอบด้วย 3 องค์ประกอบหลัก: $\text{Bias}^2$, $\text{Variance}$, และ $\text{Irreducible Error}$ ($\sigma^2$)
- **ความซับซ้อนของโมเดล (Model Complexity / Flexibility):**
  - ดีกรีต่ำ หรือ $K$ มากใน KNN $\implies$ ความซับซ้อนต่ำ $\implies$ **High Bias, Low Variance** (Underfitting)
  - ดีกรีสูง หรือ $K$ น้อยใน KNN $\implies$ ความซับซ้อนสูง $\implies$ **Low Bias, High Variance** (Overfitting)
- **กราฟ Test Error:** เป็นรูปตัว U (U-shaped Curve) จุดต่ำสุดคือจุดที่ได้ Trade-off เหมาะสมที่สุด

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **Bias-Variance Decomposition:**
  $$\mathbb{E}[(y_0 - \hat{f}(x_0))^2] = \underbrace{\left( \mathbb{E}[\hat{f}(x_0)] - f(x_0) \right)^2}_{\text{Bias}^2(\hat{f}(x_0))} + \underbrace{\mathbb{E}\left[ (\hat{f}(x_0) - \mathbb{E}[\hat{f}(x_0)])^2 \right]}_{\text{Var}(\hat{f}(x_0))} + \underbrace{\text{Var}(\epsilon)}_{\sigma^2}$$
- **Polynomial Feature Transformation (ดีกรี $d$):**
  $$\phi(x) = [x, x^2, x^3, \dots, x^d]$$

### 3. รูปแบบโค้ด Python
```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# แปลงคุณลักษณะพหุนาม
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly_train = poly.fit_transform(X_train[['x']])
X_poly_test = poly.transform(X_test[['x']])

model = LinearRegression()
model.fit(X_poly_train, y_train)
y_pred = model.predict(X_poly_test)

# การคำนวณ Bias^2 และ Variance จาก Simulation หลายรอบ
# preds shape: (n_simulations, n_test_samples)
expected_pred = np.mean(preds, axis=0)
bias_squared = np.mean((expected_pred - y_true)**2)
variance = np.mean(np.var(preds, axis=0))
```

---

## Week 7: EDA, Bootstrap CI, Independent t-test และ Chi-square

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- **Bootstrap:** การสุ่มซ้ำ (Resampling with replacement) ขนาด $n$ จากชุดข้อมูล $n$ ตัวอย่าง ใช้ประมาณช่วงความเชื่อมั่นโดยไม่ต้องสมมติการแจกแจงแบบปกติ
- **Independent Two-Sample t-test (Welch's t-test):** ทดสอบเปรียบเทียบค่าเฉลี่ยของ 2 กลุ่มที่เป็นอิสระต่อกัน (เช่น ลูกค้าสมาชิก vs ลูกค้าทั่วไป) โดยไม่สมมติว่าความแปรปรวนเท่ากัน (`equal_var=False`)
- **Chi-square Test of Independence:** ทดสอบความสัมพันธ์ระหว่างตัวแปรจัดกลุ่ม 2 ตัวแปร ($r \times c$ Contingency Table)

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **Percentile Bootstrap Confidence Interval ($1 - \alpha$):**
  $$\text{CI}_{1-\alpha} = [\hat{\theta}^*_{\alpha/2}, \; \hat{\theta}^*_{1 - \alpha/2}]$$
- **Welch's Two-Sample t-statistic:**
  $$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}, \quad s_k^2 = \frac{1}{n_k - 1} \sum_{i=1}^{n_k} (x_{ki} - \bar{x}_k)^2 \quad (ddof=1)$$
- **Chi-Square Test of Independence ($\chi^2$):**
  $$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \times C_j}{N}, \quad df = (r - 1)(c - 1)$$

### 3. รูปแบบโค้ด Python
```python
import numpy as np
import scipy.stats as stats
import pandas as pd

# 1. Percentile Bootstrap Confidence Interval
B = 2000
boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(B)]
ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

# 2. Welch's Two-Sample t-test
t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)

# 3. Chi-square Test of Independence
contingency_table = pd.crosstab(df['cat_var1'], df['cat_var2'])
chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table, correction=False)
```

---

## Week 8: Simple Linear Regression (SE, RSE, t-statistic, p-value, R²)

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- โมเดลถดถอยเชิงเส้นอย่างง่าย: $y = \beta_0 + \beta_1 x + \epsilon$
- **Residual Standard Error (RSE):** การประมาณค่าเบี่ยงเบนมาตรฐานของ $\epsilon$ (วัดขนาดความคลาดเคลื่อนโดยเฉลี่ย มีหน่วยเดียวกับ $y$)
- **Standard Error of $\hat{\beta}_1$:** วัดความไม่แน่นอนในการประมาณความชัน ยิ่งข้อมูลกระจายตัวตามแนวราบมาก ($\sum (x_i - \bar{x})^2$ มาก) ความชันยิ่งแม่นยำ (SE ยิ่งต่ำ)
- **t-statistic & p-value:** ทดสอบสมมติฐาน $H_0: \beta_1 = 0$ ว่าตัวแปร $x$ มีความสัมพันธ์เชิงเส้นกับ $y$ หรือไม่

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **OLS Estimates:**
  $$\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}, \quad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$
- **Sum of Squares:**
  $$\text{TSS} = \sum (y_i - \bar{y})^2, \quad \text{RSS} = \sum (y_i - \hat{y}_i)^2, \quad \text{ESS} = \sum (\hat{y}_i - \bar{y})^2$$
- **Residual Standard Error (RSE):**
  $$\text{RSE} = \sqrt{\frac{\text{RSS}}{n - 2}}$$
- **Standard Error of Slope:**
  $$\text{SE}(\hat{\beta}_1) = \frac{\text{RSE}}{\sqrt{\sum (x_i - \bar{x})^2}}$$
- **t-statistic for $\hat{\beta}_1$ ($df = n - 2$):**
  $$t = \frac{\hat{\beta}_1 - 0}{\text{SE}(\hat{\beta}_1)}$$
- **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = \frac{\text{ESS}}{\text{TSS}} = r_{xy}^2$$

### 3. รูปแบบโค้ด Python
```python
import statsmodels.formula.api as smf

model = smf.ols('y ~ x', data=df).fit()

beta_0 = model.params['Intercept']
beta_1 = model.params['x']
se_beta_1 = model.bse['x']
t_stat = model.tvalues['x']
p_val = model.pvalues['x']
r_squared = model.rsquared
rse = np.sqrt(model.scale)  # หรือ np.sqrt(model.ssr / model.df_resid)
```

---

## Week 9: Multiple Linear Regression (Adjusted R², AIC, BIC, CI vs PI)

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- การถดถอยเชิงเส้นพหุคูณควบคุมผลกระทบจากตัวแปรสับสน (Confounding Variables)
- **$R^2$ vs Adjusted $R^2$:** $R^2$ ปกติจะเพิ่มขึ้นเสมอเมื่อใส่ตัวแปรเพิ่มเข้าไป แม้ตัวแปรนั้นจะไร้ประโยชน์ แต่ Adjusted $R^2$ จะลงโทษตามจำนวนตัวแปรที่เพิ่มขึ้น
- **AIC / BIC:** เกณฑ์คัดเลือกโมเดล (Information Criteria) ค่ายิ่งต่ำยิ่งดี โดย BIC ลงโทษความซับซ้อนมากกว่า AIC เมื่อ $n \ge 8$
- **Confidence Interval (CI) vs Prediction Interval (PI):**
  - CI: ช่วงความเชื่อมั่นของ **ค่าเฉลี่ยประชากร** $E[y \mid x_0]$ (แคบกว่า)
  - PI: ช่วงการทำนายของ **ลูกค้ารายบุคคล** $y_0$ (กว้างกว่า เพราะต้องรวมความแปรปรวนสุ่ม $\epsilon$)

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **Matrix OLS:**
  $$\hat{\beta} = (X^\top X)^{-1} X^\top y, \quad \text{Cov}(\hat{\beta}) = \sigma^2 (X^\top X)^{-1}$$
- **Adjusted $R^2$ ($p$ คือจำนวนตัวแปรอิสระ):**
  $$R_{\text{adj}}^2 = 1 - \frac{\text{RSS} / (n - p - 1)}{\text{TSS} / (n - 1)} = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}$$
- **Akaike Information Criterion (AIC) & Bayesian Information Criterion (BIC):**
  $$\text{AIC} = n \ln\left(\frac{\text{RSS}}{n}\right) + 2(p + 1) + C$$
  $$\text{BIC} = n \ln\left(\frac{\text{RSS}}{n}\right) + (p + 1)\ln(n) + C$$
- **CI vs PI Standard Error:**
  $$\text{SE}(\text{CI}) = \hat{\sigma} \sqrt{x_0^\top (X^\top X)^{-1} x_0}$$
  $$\text{SE}(\text{PI}) = \hat{\sigma} \sqrt{1 + x_0^\top (X^\top X)^{-1} x_0} = \sqrt{\hat{\sigma}^2 + \text{SE}(\text{CI})^2}$$

### 3. รูปแบบโค้ด Python
```python
import statsmodels.formula.api as smf

model = smf.ols('spend ~ income + tenure + visits', data=df).fit()

adj_r2 = model.rsquared_adj
aic = model.aic
bic = model.bic
f_stat = model.fvalue

# การคำนวณ CI และ PI สำหรับข้อมูลใหม่
new_obs = pd.DataFrame({'income': [50.0], 'tenure': [24], 'visits': [10]})
pred_res = model.get_prediction(new_obs)
ci_table = pred_res.summary_frame(alpha=0.05)

ci_lower = ci_table['mean_ci_lower'].iloc[0]
ci_upper = ci_table['mean_ci_upper'].iloc[0]
pi_lower = ci_table['obs_ci_lower'].iloc[0]
pi_upper = ci_table['obs_ci_upper'].iloc[0]
```

---

## Week 10: Dummy Variables, Interaction Terms, Diagnostics และ VIF

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- **Dummy Variables:** ตัวแปรกลุ่มที่มี $K$ หมวดหมู่ จะต้องสร้างตัวแปรหุ่น $K - 1$ ตัว โดยหมวดหมู่ที่เหลือเป็นฐานเปรียบเทียบ (Reference Category)
- **Interaction Terms ($x_1 \times x_2$):** ปฏิสัมพันธ์หมายถึง "ผลกระทบของ $x_1$ ต่อ $y$ ขึ้นอยู่กับระดับของ $x_2$"
- **Multicollinearity & VIF:** เมื่อตัวแปรอิสระทำนายกันเองได้ดี จะทำให้ Standard Error ของสัมประสิทธิ์บวมโต ค่า $t$ ลดลง และโมเดลไม่เสถียร หาก $\text{VIF} > 5$ หรือ $10$ แสดงว่ามีปัญหา Multicollinearity รุนแรง

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **โมเดลที่มี Interaction:**
  $$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 (x_1 x_2)$$
  $$\frac{\partial \hat{y}}{\partial x_1} = \beta_1 + \beta_3 x_2$$
- **Variance Inflation Factor (VIF):**
  $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
  โดย $R_j^2$ คือสัมประสิทธิ์การตัดสินใจจากการนำตัวแปร $x_j$ ไปถดถอยกับตัวแปรอิสระที่เหลือทั้งหมด

### 3. รูปแบบโค้ด Python
```python
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Regression พร้อม Dummy (ระบุ Reference) และ Interaction
model = smf.ols('spend ~ C(tier, Treatment(reference="Silver")) + income * web_visits', data=df).fit()

# 2. การคำนวณ VIF สำหรับตัวแปรเชิงปริมาณ
features = ['income', 'tenure', 'web_visits']
X_mat = df[features].copy()
X_mat['Intercept'] = 1.0  # ต้องมี Intercept เสมอ

vifs = {feat: variance_inflation_factor(X_mat.values, i) 
        for i, feat in enumerate(X_mat.columns) if feat != 'Intercept'}
```

---

## Week 11: Logistic Regression ธุรกิจ (Odds Ratio, Confusion Matrix, Threshold, Class Weight)

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- ใช้สำหรับตัวแปรตอบสนองที่เป็น Binary ($y \in \{0, 1\}$) โดยประมาณค่าความน่าจะเป็นผ่านฟังก์ชัน Sigmoid (Logit Link)
- **Odds vs Probability:** แต้มต่อ $\text{Odds} = \frac{p}{1-p}$
- **Odds Ratio ($\text{OR} = e^{\beta_j}$):** เมื่อ $x_j$ เพิ่มขึ้น 1 หน่วย แต้มต่อจะเพิ่มขึ้นเป็น $e^{\beta_j}$ เท่า
- **Threshold Tuning & Class Weight:** ในโจทย์ธุรกิจที่มี Class Imbalance (เช่น Churn มีน้อย) การตัดที่ Threshold $0.5$ อาจจับลูกค้าที่เลิกซื้อไม่ได้ การลด Threshold หรือใช้ `class_weight='balanced'` ช่วยเพิ่ม Recall ของคลาส 1 ได้

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **Logit (Log-Odds):**
  $$z = \ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p$$
- **Probability:**
  $$p(x) = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$
- **Odds Ratio:**
  $$\text{OR}_j = e^{\beta_j}$$
- **Confusion Matrix Metrics:**
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall (Sensitivity)} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}, \quad \text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Balanced Class Weights:**
  $$w_0 = \frac{n}{2 n_0}, \quad w_1 = \frac{n}{2 n_1}$$

### 3. รูปแบบโค้ด Python
```python
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# 1. statsmodels (Maximum Likelihood, ไม่มี Regularization)
logit_mod = smf.logit('churn ~ tenure + spend_k + web_visits', data=df).fit()
odds_ratios = np.exp(logit_mod.params)
probs = logit_mod.predict(df)

# 2. scikit-learn พร้อม Balanced Class Weight
lr_clf = LogisticRegression(penalty=None, class_weight='balanced')
lr_clf.fit(X_train, y_train)

# 3. กำหนด Threshold เอง
custom_threshold = 0.35
y_pred_custom = (lr_clf.predict_proba(X_test)[:, 1] >= custom_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_custom)
```

---

## Week 12: LDA, QDA, Gaussian Naive Bayes และ KNN ธุรกิจ

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- **Generative Models (LDA, QDA, Naive Bayes):** โมเดลการแจกแจงร่วม $P(X \mid Y=k)$ แล้วใช้ทฤษฎีบทเบย์สหา Posterior $P(Y=k \mid X)$
- **LDA (Linear Discriminant Analysis):** สมมติว่าทุกคลาสมีการแจกแจงแบบปกติหลายตัวแปร และมี **Covariance Matrix เท่ากันร่วมกัน** ($\Sigma_k = \Sigma$) ทำให้เส้นแบ่งการตัดสินใจเป็นเส้นตรง
- **QDA (Quadratic Discriminant Analysis):** แต่ละคลาสมี **Covariance Matrix แยกเฉพาะตัว** ($\Sigma_k$) เส้นแบ่งเป็นรูปโค้งกำลังสอง
- **Gaussian Naive Bayes (GNB):** สมมติว่าตัวแปรอิสระต่อกันโดยสิ้นเชิงเมื่อทราบคลาส ($\Sigma_k$ เป็น Diagonal Matrix)

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **Bayes' Theorem for Classification:**
  $$P(Y=k \mid X=x) = \frac{\pi_k f_k(x)}{\sum_{l=1}^K \pi_l f_l(x)}$$
- **LDA Linear Discriminant Function:**
  $$\delta_k(x) = x^\top \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^\top \Sigma^{-1} \mu_k + \ln(\pi_k)$$
- **QDA Quadratic Discriminant Function:**
  $$\delta_k(x) = -\frac{1}{2} \ln |\Sigma_k| - \frac{1}{2} (x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k) + \ln(\pi_k)$$
- **Gaussian Naive Bayes Likelihood:**
  $$f_k(x) = \prod_{j=1}^p \frac{1}{\sqrt{2\pi \sigma_{kj}^2}} \exp\left( -\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2} \right)$$

### 3. รูปแบบโค้ด Python
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
gnb = GaussianNB().fit(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors=9).fit(X_train_scaled, y_train)

# ดึงค่าความน่าจะเป็นคลาส 1
prob_lda = lda.predict_proba(X_test)[:, 1]
```

---

## Week 13: GLM (Poisson Regression), ROC AUC, Class Imbalance

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- **Poisson Regression:** ออกแบบมาสำหรับข้อมูลการนับ (Count Data: $Y \in \{0, 1, 2, \dots\}$) โดยใช้ Log Link Function เพื่อรับประกันว่าค่าคาดหมาย $\lambda > 0$ เสมอ
- **Incidence Rate Ratio ($\text{IRR} = e^{\beta_j}$):** สัดส่วนการเปลี่ยนแปลงของความถี่เหตุการณ์เมื่อตัวแปร $x_j$ เพิ่มขึ้น 1 หน่วย
- **ROC Curve & AUC:** กราฟความสัมพันธ์ระหว่าง Sensitivity (TPR) กับ $1 - \text{Specificity}$ (FPR) ที่ทุกระดับ Threshold ค่า AUC คือความน่าจะเป็นที่โมเดลจะให้ Score ตัวอย่างคลาสบวกสูงกว่าตัวอย่างคลาสลบ

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **Poisson Model (Log Link):**
  $$Y \sim \text{Poisson}(\lambda), \quad \ln(\lambda) = \beta_0 + \sum_{j=1}^p \beta_j x_j \implies \lambda = \mathbb{E}[Y \mid x] = \exp\left(\beta_0 + \sum_{j=1}^p \beta_j x_j\right)$$
- **Incidence Rate Ratio (IRR):**
  $$\text{IRR}_j = \exp(\beta_j) = \frac{\mathbb{E}[Y \mid x_j + 1]}{\mathbb{E}[Y \mid x_j]}$$
- **ROC Curve Coordinates:**
  $$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \text{Sensitivity}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = 1 - \text{Specificity}$$
- **AUC (Area Under ROC Curve):**
  $$\text{AUC} = P(\hat{p}_1 > \hat{p}_0)$$

### 3. รูปแบบโค้ด Python
```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, roc_curve

# 1. Poisson Regression
poisson_mod = smf.glm('order_count ~ income + web_visits', 
                      data=df, family=sm.families.Poisson()).fit()
irr = np.exp(poisson_mod.params)
predicted_counts = poisson_mod.predict(df)

# 2. ROC AUC Score & Curve
auc_score = roc_auc_score(y_true, y_pred_prob)
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
```

---

## Week 14: Cross-Validation (Validation Set, LOOCV, K-Fold, 1-SE Rule)

### 1. สิ่งที่ควรรู้ & แนวคิดหลัก
- การประเมิน Generalization Error เพื่อป้องกันการประเมินที่มองโลกในแง่ดีเกินจริง (Optimistic Bias)
- **ระเบียบวิธี Cross-Validation:**
  - **Validation Set Approach:** แบ่งคงที่ (เช่น คู่/คี่ หรือ 70:30) เร็วแต่มีความแปรปรวนสูง
  - **Leave-One-Out CV (LOOCV):** กันตัวอย่างทีละ 1 จุด ($K=n$) แทบไม่มี Bias แต่ต้นทุนการคำนวณสูง
  - **$K$-Fold CV ($K=5$ หรือ $10$):** สมดุล Bias-Variance ดีที่สุด นิยมใช้มากที่สุดในทางปฏิบัติ
  - **Stratified $K$-Fold CV:** รักษาสัดส่วนคลาสในแต่ละ Fold ให้เท่ากับประชากร เหมาะอย่างยิ่งกับ Imbalanced Data
- **One-Standard-Error Rule (1-SE Rule):** เลือกโมเดลที่เรียบง่ายที่สุด (Parsimonious Model เช่น ดีกรีต่ำสุด) ที่มี $\text{CV MSE} \le \text{CV MSE}_{\min} + \text{SE}_{\min}$ เพื่อลดความเสี่ยง Overfitting

### 2. สูตรคณิตศาสตร์ (รูปแบบปกติ)
- **$K$-Fold Cross-Validation Error:**
  $$\text{CV}_{(K)} = \frac{1}{K} \sum_{k=1}^K \text{MSE}_k, \quad \text{MSE}_k = \frac{1}{n_k} \sum_{i \in \mathcal{C}_k} (y_i - \hat{y}_i^{(-k)})^2$$
- **Standard Error ของ $K$-Fold CV:**
  $$\text{SE} = \frac{\text{SD}(\text{MSE}_1, \dots, \text{MSE}_K, \; ddof=1)}{\sqrt{K}} = \frac{1}{\sqrt{K}} \sqrt{\frac{1}{K-1} \sum_{k=1}^K (\text{MSE}_k - \text{CV}_{(K)})^2}$$
- **One-Standard-Error Threshold:**
  $$\text{Threshold} = \text{CV MSE}_{\min} + \text{SE}_{\min}$$
  $$d^* = \min \{ d \mid \text{CV MSE}(d) \le \text{Threshold} \}$$
- **Generalization Gap:**
  $$\Delta_{\text{Gen}} = \overline{\text{MSE}}_{\text{val}} - \overline{\text{MSE}}_{\text{train}}$$

### 3. รูปแบบโค้ด Python
```python
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, f1_score

# 1. การวนลูป K-Fold CV พร้อมป้องกัน Data Leakage
kf = KFold(n_splits=5, shuffle=False)
fold_mses = []

for train_idx, val_idx in kf.split(X):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[train_idx])
    X_val_s = scaler.transform(X[val_idx])  # ใช้ scaler ที่ fit จาก train เท่านั้น
    
    model.fit(X_train_s, y[train_idx])
    pred_val = model.predict(X_val_s)
    fold_mses.append(mean_squared_error(y[val_idx], pred_val))

cv_mse = np.mean(fold_mses)

# 2. การคำนวณ One-Standard-Error Rule
best_deg = min(cv_scores, key=cv_scores.get)
se_best = np.std(fold_scores[best_deg], ddof=1) / np.sqrt(5)
threshold = cv_scores[best_deg] + se_best
chosen_deg = min([d for d in [1, 2, 3] if cv_scores[d] <= threshold])
```

---

## ตารางสรุป Quick Reference ฟังก์ชัน Python ที่ใช้บ่อยที่สุด

| สัปดาห์ | หัวข้อหลัก | ฟังก์ชัน / Method สำคัญใน Python | Library |
| :---: | :--- | :--- | :--- |
| **Week 1** | Vector Norms & Angles | `np.linalg.norm(x, ord=1/2/inf)`, `np.dot(x, y)` | NumPy |
| **Week 2** | Matrices & Linear Systems | `A @ B`, `np.linalg.inv(A)`, `np.linalg.solve(A, b)`, `np.linalg.det(A)` | NumPy |
| **Week 3** | Rank, Projection & OLS | `np.linalg.matrix_rank(A)`, `scipy.linalg.null_space(A)`, `np.linalg.cond(A)` | NumPy / SciPy |
| **Week 4** | Eigen, SVD & PCA | `np.linalg.eig(A)`, `np.linalg.svd(A)`, `PCA(n_components=k)` | NumPy / sklearn |
| **Week 5** | Regression & Scaling | `StandardScaler()`, `LinearRegression()`, `KNeighborsRegressor()` | sklearn |
| **Week 6** | Bias-Variance Trade-Off | `PolynomialFeatures(degree=d, include_bias=False)` | sklearn |
| **Week 7** | Bootstrap & Hypothesis Tests | `stats.ttest_ind(equal_var=False)`, `stats.chi2_contingency(correction=False)` | SciPy |
| **Week 8** | Simple Linear Regression | `smf.ols('y ~ x', data=df).fit()`, `model.bse`, `model.tvalues`, `model.pvalues` | statsmodels |
| **Week 9** | Multiple Linear Regression | `model.rsquared_adj`, `model.aic`, `model.bic`, `model.get_prediction()` | statsmodels |
| **Week 10** | Dummy, Interaction, VIF | `C(var, Treatment(ref))`, `x1 * x2`, `variance_inflation_factor(X, i)` | statsmodels |
| **Week 11** | Logistic Regression | `smf.logit().fit()`, `LogisticRegression(class_weight='balanced')` | statsmodels / sklearn |
| **Week 12** | Discriminant & Naive Bayes | `LinearDiscriminantAnalysis()`, `QuadraticDiscriminantAnalysis()`, `GaussianNB()` | sklearn |
| **Week 13** | Count Data & ROC/AUC | `smf.glm(family=sm.families.Poisson())`, `roc_auc_score()`, `roc_curve()` | statsmodels / sklearn |
| **Week 14** | Resampling & CV | `KFold(shuffle=False)`, `StratifiedKFold(shuffle=False)`, `LeaveOneOut()` | sklearn |
#   M a t h - b y - I t h c h i  
 