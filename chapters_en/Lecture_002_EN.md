# Lecture 2 | Ink into Stars: Raw Materials — Data, Features, and Labels

> *"When a master jade carver picks up a piece of raw stone, he never rushes to make the first cut. First, he reads the stone—studying the grain, tracing the hidden fractures, seeing where the color runs. In machine learning, the model is the carving knife; the data is the raw stone."*
> — **Ink into Stars: 108 Lectures on Machine Learning (Lecture 2)**

In [Lecture 1](https://youtu.be/uc1q-uyDsrw), we learned how to hold the carving knife: we took Ronald Fisher's 150 iris flowers, measured Euclidean distance in four-dimensional space, and reached **96.67% accuracy** in four lines of Python.

Yet in real-world engineering, the sharpest algorithm in the world will fail if the raw stone is flawed. Before we explore more complex models, we must learn the craft of reading the stone—understanding the raw materials of machine learning: **Data, Features, and Labels**.

---

## 1. One Table, Three Core Words: Sample, Feature, and Label

Suppose we want to predict the transaction price of second-hand apartments and collect a simple ledger:

| Area ($\text{m}^2$) | Bedrooms | Age (Years) | District | Near Subway | Price (\$10k) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 89 | 2 | 12 | East | Yes | 520 |
| 126 | 3 | 5 | West | No | 760 |
| 58 | 1 | 23 | East | Yes | 310 |
| 142 | 4 | 8 | South | No | 640 |

Hidden inside this table are the three foundational nouns of supervised learning:
1. **Sample (or Instance)**: Every **row** is one apartment—one individual item we observe.
2. **Feature**: Every descriptive **column**—Area, Bedrooms, Age, District, Near Subway—is a property used to characterize the sample.
3. **Label (or Target)**: The final **column**—Transaction Price—is the answer we want the machine to predict.

> **One-sentence rule:** *One row is one sample, one column is one feature, and the final column is the answer.*

### From Table to Linear Algebra: $\boldsymbol{X}$ and $\boldsymbol{y}$

When we hand this table over to mathematics, it becomes two objects (after encoding categories into numbers):
- **The Feature Matrix $\boldsymbol{X} \in \mathbb{R}^{n \times m}$**: All $n$ samples and $m$ features arranged in an $n$-by-$m$ grid. In Lecture 1's Iris dataset, $\boldsymbol{X}$ was a $150 \times 4$ matrix.
- **The Label Vector $\boldsymbol{y} = (y_1, y_2, \dots, y_n)^{\top}$**: All $n$ target answers stacked into a column.
- **The Feature Vector $\boldsymbol{x}_i = (x_{i1}, x_{i2}, \dots, x_{im})^{\top}$**: All $m$ features of the $i$-th sample, where $x_{ij}$ denotes the $j$-th feature of the $i$-th sample.

### The Geometry of Choice: Feature Space

Look at the table from a geometric perspective:
- With **2 features** (Area and Age), every apartment is a point on a 2D plane.
- With **3 features**, every apartment is a point in 3D space.
- With **$m$ features**, every sample is a single drop of ink suspended in an **$m$-dimensional feature space**.

This brings us back to the central motif of our series: **Every sample is a drop of ink in feature space.** Choosing which features to measure determines where those drops of ink land and how far apart they sit. Choose informative features, and samples of the same kind naturally gather into clear constellations; choose poor features, and the ink blurs into an inseparable tangle where even the smartest algorithm is helpless.

---

## 2. Turning Everything into Numbers

Models only understand numbers. When we inspect a feature in an old apothecary shop, we encounter two fundamentally different kinds of measurements:

1. **Numerical Features**: Like herbs weighed on a brass steelyard balance—three錢, five錢, or an apartment's floor area ($89.5\text{ m}^2$) and bedroom count ($2$). They are born as numbers; you can add, subtract, and compare their magnitudes.
2. **Categorical Features**: Like the handwritten herb names on the hundred-drawer cabinet—*Angelica*, *Astragalus*, *Licorice*. They are names, not quantities.
   - **Ordinal Categories** have a natural rank: *High School $\to$ Bachelor's $\to$ Master's*. Encoding them as $0, 1, 2$ preserves their order (though equal numeric steps don't always imply equal real-world gaps).
   - **Nominal Categories** have **no order at all**: *East District, West District, South District*, or *Beijing, Shanghai, Shenzhen*.

### The Trap of Integer Encoding vs. One-Hot Encoding

What happens if we encode nominal categories as integers—say, $\text{East} = 1$, $\text{West} = 2$, $\text{South} = 3$—and feed them into $k$-Nearest Neighbors?
- The model will hallucinate that South ($3$) is "larger" than East ($1$).
- Worse, it will believe the distance from East to South ($|1 - 3| = 2$) is **twice as far** as the distance from East to West ($|1 - 2| = 1$). We have fabricated a false geometry out of thin air.

The principled solution is **One-Hot Encoding**. For three categories, we create three binary switches:

$$\text{East} = (1, 0, 0), \qquad \text{West} = (0, 1, 0), \qquad \text{South} = (0, 0, 1)$$

Now calculate the Euclidean distance between any two distinct districts:

$$\|\boldsymbol{e}_a - \boldsymbol{e}_b\| = \sqrt{1^2 + (-1)^2} = \sqrt{2}$$

Three lanterns arranged at the vertices of an equilateral triangle—every pair separated by $\sqrt{2}$, none greater or closer than any other.

---

## 3. Unstructured Data: Images, Text, and Sound

Most data in the wild does not arrive in neat spreadsheet columns. Photographs, essays, and audio recordings are **unstructured data**. How do we turn them into vectors?

- **Images = Grids of Pixels**: An $8 \times 8$ grayscale handwritten digit is 64 brightness values (from $0$ to $16$). Unroll the 8 rows end-to-end, and the image becomes a **64-dimensional vector**. A $1920 \times 1080$ RGB color photograph has three channels (Red, Green, Blue), yielding $1920 \times 1080 \times 3 = 6,220,800$ numbers.
- **Text = Counts of Words (Bag-of-Words)**: Like sorting movable type into a cloth pouch, we list a vocabulary of unique words and count how many times each word appears in a sentence:
  - `"I love machine learning"` $\to [1, 1, 1, 1, 0]$
  - Simple and effective—but once words are poured into a bag, **word order is lost**. *"Dog bites man"* and *"Man bites dog"* produce the exact same vector. Teaching machines to understand sequence and meaning will be a major arc of the second half of this course.
- **Audio = Sampled Vibrations**: Sound is air pressure vibrating over time. A CD-quality microphone measures amplitude **44,100 times per second** ($44.1\text{ kHz}$). One second of mono audio is a vector of 44,100 numbers.

> **Whether table, image, text, or sound: in the eyes of a machine, all data ultimately becomes a vector of numbers.**

---

## 4. Where Labels Come From—and Why Defining the Label Defines the Problem

Supervised learning requires labels, and labels never fall from the sky. They come from three sources, each with its own trade-off:
1. **Human Annotation**: Hiring annotators to label images or text one by one. High quality, but expensive, slow, and subject to inter-annotator disagreement.
2. **Natural Business Outcomes**: Whether a borrower defaulted on a loan, or whether a user clicked an ad. Cheap and abundant, but requires waiting—sometimes a full year for a loan to mature.
3. **Rule-Based Weak Supervision**: Automatically labeling 1-star reviews as "Negative." Instant, but noisy.

Even more critical is the **Label Definition (and Time Horizon)**. "Loan default" sounds unambiguous—until you have to write the SQL query: *Does 1 day past due count as default, or 90 days past due?* Change the definition, and the labels change; change the labels, and the model is solving a different problem.

> **To define the label is to define the problem. If the label is defined poorly, every subsequent optimization is merely solving the wrong problem with higher precision.**

---

## 5. Training, Validation, and the Sealed Exam Paper

In Lecture 1, we split data into two parts: Training and Test. Why is a two-way split not enough in real projects?

Recall the hyperparameter $K$ in $k$-Nearest Neighbors. If we try $K \in \{1, 3, 5, 7, 9, 15\}$, check which $K$ gets the highest score on the **test set**, and report that winning score, we have cheated: **the test set is no longer unseen**.

Why does peeking inflate the score? A model's empirical error rate on a finite dataset $D$ is:

$$\hat{E}(f; D) = \frac{1}{|D|} \sum_{(\boldsymbol{x}_i, y_i) \in D} \mathbb{I}\big(f(\boldsymbol{x}_i) \neq y_i\big)$$

Because $|D| = n$ is finite, every measured score carries random sampling noise with standard error:

$$\mathrm{SE} = \sqrt{\frac{p(1-p)}{n}}$$

On a small test set of $n = 36$ samples around $p \approx 0.64$, $\widehat{\mathrm{SE}} \approx 0.080$, meaning a 95% confidence interval spans roughly $\pm 16$ percentage points! When you evaluate many candidate models and pick the maximum score, you systematically select the candidate that had **both decent skill and good luck** on that split. **A selected score is inherently optimistic.**

Therefore, we must split our data into **three parts**:
1. **Training Set (The Workbook)**: Used to fit parameters.
2. **Validation Set (The Mock Exam)**: Used to tune hyperparameters (like $K$) and choose feature pipelines.
3. **Test Set (The Final Imperial Exam)**: Like the *húmíng* (糊名) practice in ancient Chinese imperial examinations—where candidates' names were sealed under paper strips so examiners could grade blind—the test set is locked in a drawer and opened **only once** after all decisions are final.

---

## 6. Big Data $\neq$ Good Data: The 1936 *Literary Digest* Disaster

Before splitting data, we must ask where the data came from. In 1936, the American magazine *Literary Digest* mailed out **10 million** mock ballots and collected **2.4 million** responses—a massive sample—predicting a landslide victory for Alf Landon. Instead, Franklin D. Roosevelt carried 46 of 48 states.

How could 2.4 million samples fail so catastrophically?
1. **Sampling Bias**: The mailing list was drawn heavily from telephone directories and automobile registrations—luxuries owned disproportionately by the wealthy during the Great Depression.
2. **Non-Response Bias**: Those who bothered to mail back postcards were a self-selected subset of voters.

**More samples do not mean right samples.** And when splitting data, we must respect real-world boundaries:
- **Temporal Split**: If predicting tomorrow, train on the past and validate/test on the future.
- **Group Split**: If multiple medical scans belong to the same patient and the goal is diagnosing new patients, keep all scans of a single patient in the same split so the model cannot memorize patient identity across training and test sets.

---

## 7. Hands-On Python: From 63.9% to 94.4% on the Italian Wine Dataset

Let's test these principles on the classic **UCI Wine Dataset**: $178$ bottles of wine from the same region of Italy, fermented from $3$ grape cultivars, each described by $13$ chemical measurements (Alcohol, Malic Acid, Magnesium, Color Intensity, Proline, etc.).

### Step 1: Two Cuts (`train_test_split`) on Raw Features

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_wine(return_X_y=True)

# First cut: lock 20% (36 bottles) in the drawer as the Test Set
X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Second cut: take 25% of the remaining 80% (36 bottles) as the Validation Set
X_train, X_val, y_train, y_val = train_test_split(
    X_rest, y_rest, test_size=0.25, random_state=0, stratify=y_rest
)

for k in [1, 3, 5, 7, 9, 15]:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    print("K =", k, "Validation:", round(model.score(X_val, y_val), 3))

final = KNeighborsClassifier(n_neighbors=1).fit(X_rest, y_rest)
print("Test Set:", round(final.score(X_test, y_test), 3))
```

Output:
```text
K = 1 Validation: 0.833
K = 3 Validation: 0.778
K = 5 Validation: 0.75
K = 7 Validation: 0.722
K = 9 Validation: 0.806
K = 15 Validation: 0.75
Test Set: 0.639
```

Look at what happened: $K=1$ achieved **83.3%** on the validation set, yet when we unlocked the test set drawer, accuracy dropped nearly 20 points to **63.9%** ($23/36$)!

### Step 2: Why Did $k$-NN Struggle? Feature Scale & Standardization

Why was 63.9% so low? Inspect the 13 chemical features: **Proline** ranges from $278$ to $1,680$, while **Nonflavanoid Phenols** ranges from $0.13$ to $0.66$. When $k$-NN computes Euclidean distance, Proline alone dominates the sum of squares, drowning out the other 12 chemical indicators!

To give every feature an equal voice, we **standardize** each feature to zero mean and unit standard deviation:

$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$

Crucially, **where do the ruler's mean $\mu_j$ and standard deviation $\sigma_j$ come from? Only from the training set!** Measuring $\mu_j$ and $\sigma_j$ on the full dataset before splitting is a classic form of data leakage. Using `make_pipeline`, we bind `StandardScaler` and `KNeighborsClassifier` together so the ruler is strictly fit on training data:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

for k in [1, 3, 5, 7, 9, 15]:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    pipe.fit(X_train, y_train)
    print("K =", k, "Scaled Validation:", round(pipe.score(X_val, y_val), 3))

final_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
final_pipe.fit(X_rest, y_rest)
print("Scaled Test Set:", round(final_pipe.score(X_test, y_test), 3))
```

With standardization, validation selects $K = 5$, and test accuracy leaps from **63.9% ($23/36$)** to **94.4% ($34/36$)**!

We didn't change the algorithm at all—we simply respected the raw stone and aligned our rulers. **Data and features determine the upper bound of machine learning.**

---

## Three Takeaways from Lecture 2

1. **One row is one sample, one column is one feature, and the label is the answer to predict.** In the eyes of a machine, all data—tables, images, text, and sound—ultimately becomes a vector.
2. **To define the label is to define the problem.**
3. **Fit parameters on the training set, choose models on the validation set, and open the test set only once.**

*In Lecture 3 ("Three Waves: A Brief History of Machine Learning"), we step back across seventy years of history to see how humanity walked three distinct paths—and survived two AI winters—in the quest to make machines learn.*
