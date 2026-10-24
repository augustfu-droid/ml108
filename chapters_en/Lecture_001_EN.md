# Lecture 1 | Ink into Stars: What Is Machine Learning?

*Series: **Ink into Stars: 108 Lectures on Machine Learning** (`墨点成星：机器学习一百零八讲`)*

> *Every data point is a drop of ink on rice paper. Machine learning is the art of seeing constellations in ten thousand dots of ink.*

---

## Introduction

Thousands of years ago, someone looked up at the night sky. All they saw at first were scattered points of light. Yet after watching night after night, they began drawing invisible lines between the stars: the Big Dipper, Orion, Scorpius. The stars themselves never changed. What changed was the observer—who, through countless nights of observation, **learned to see patterns**.

Today, we hand that same task to machines. We give a computer thousands—or billions—of "points of light" (data), and ask it to discover the underlying constellations on its own, then use those patterns to make judgments about situations it has never seen before.

In this opening lecture, we answer four foundational questions:

1. What is the fundamental difference between **Machine Learning** and **Traditional Programming**?
2. Can the everyday word **"learning"** be defined with mathematical precision?
3. What are the **three primary paradigms** of machine learning?
4. What does it actually look like to **train your first model** from scratch?

---

## 1.1 Two Ways to Get a Machine to Act

### Approach 1: Rules — Humans Tell the Machine What to Do

Suppose you want to write a program that automatically filters out spam emails. The most direct approach is to sit down and write out the rules yourself:

- If an email contains phrases like *"You won a prize"*, *"Claim for free"*, or *"Click this link"*, flag it as spam.
- If the sender is not in the user's address book and the email contains more than five exclamation marks, flag it as spam.

This is **Traditional Programming**: humans figure out the rules, translate them into explicit code, and the machine executes them faithfully.

$$\text{Rules} + \text{Data} \;\longrightarrow\; \text{Answers}$$

When the rules of a problem are clear-cut—such as calculating personal income tax under a tax code, or sorting a list of numbers—traditional programming is unbeatable. Yet in the real world, rule-writing quickly hits three walls:

1. **The rules are endless:** As soon as you block `"prize"`, spammers write `"p r i z e"` or `"pr1ze"`. Hand-crafted rules can never keep up with a shifting adversary.
2. **The rules are ineffable:** Try writing explicit `if-else` pixel rules to recognize whether a photograph contains a cat. What angle are the ears? How many whiskers are visible? What if the cat is curled up inside a cardboard box? You can recognize the cat instantly, yet your pen freezes the moment you try to reduce your intuition to explicit conditions.
3. **The rules collide:** Once thousands of hand-written exceptions accumulate, patching one loophole breaks three others, until no one dares touch the codebase.

### Approach 2: Examples — The Machine Finds the Rules

Machine learning flips the paradigm: **instead of telling the machine the rules, we feed it a large collection of examples and let it discover the rules from data.**

We collect 10,000 historical emails, each labeled `"Spam"` or `"Normal"`, and hand them to a learning algorithm. From these labeled examples, the algorithm distills a decision-making tool—which we call a **Model**. When a new email arrives, the model evaluates it and outputs a prediction.

$$\text{Data} + \text{Answers} \;\longrightarrow\; \text{Rules (Model)}$$

Putting the two equations side by side reveals the core inversion at the heart of machine learning: **in traditional programming, humans write the rules; in machine learning, machines learn the rules from data.**

### A Brief Glimpse of History

The term *"Machine Learning"* was popularized around 1959 by the American computer scientist **Arthur Samuel** at IBM. Samuel wrote a program that played checkers. By playing game after game—including against itself—the program accumulated experience, refined how it evaluated board positions, and eventually learned to beat Samuel himself.

A program playing better than the person who wrote it—familiar as that sounds today—was a revolutionary realization in 1959: **a machine's capability does not have to be bounded by the explicit rules its creator knows how to write down.**

---

## 1.2 What Is "Learning"? A Precise Definition

"Learning" is an everyday word, but science demands a testable definition. In his classic 1997 textbook *Machine Learning*, **Tom Mitchell** at Carnegie Mellon University gave the definition still used across the field today:

> *A computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$.*

Three letters, three checkable components:

| Component | Meaning | Spam Filter Example |
|---|---|---|
| **Task $T$** | What the system is trying to do | Classify whether an incoming email is spam or normal |
| **Experience $E$** | What data the system learns from | 10,000 historical emails labeled as spam or normal |
| **Performance $P$** | How we measure success | The percentage of emails classified correctly (accuracy) |

Notice the crux of Mitchell's definition: **performance $P$ must improve as experience $E$ grows** (provided the data and metric are properly chosen—if 99 out of 100 emails are normal, a lazy program that lets every email through achieves 99% accuracy while guarding nothing!). A hard-coded program performs the same no matter how much data it sees; a learning system sharpens its judgment through experience.

### Saying It in the Language of Mathematics

Now let us express this idea in mathematical notation for the first time. It takes only three lines.

Denote the input features of each example as $\boldsymbol{x}$ (for instance, measurements or word counts of an email) and its true target answer as $y$. Across $n$ historical examples, our dataset is:

$$D = \{(\boldsymbol{x}_1, y_1),\ (\boldsymbol{x}_2, y_2),\ \dots,\ (\boldsymbol{x}_n, y_n)\}$$

1. **The Model:** We want a function $f$ that takes an input $\boldsymbol{x}$ and outputs a prediction $\hat{y}$ (pronounced *"y-hat"*—the little hat always denotes a predicted value):

$$\hat{y} = f(\boldsymbol{x};\ \boldsymbol{\theta})$$

Here $\boldsymbol{\theta}$ (*theta*) represents the **parameters** of the model—think of them as a row of brass tuning knobs inside the machine. Turn the knobs to different positions, and the model makes different predictions.

2. **The Loss:** How far off is a prediction $\hat{y}$ from the true answer $y$? Imagine standing at an archery range: the bullseye is the true answer $y$, and the spot where your arrow lands is the prediction $\hat{y}$. We define a **loss function** $L(\hat{y}, y)$—such as squared error $(\hat{y} - y)^2$—that penalizes larger misses more heavily. Averaging the loss over all $n$ training examples gives the model's overall report card $J(\boldsymbol{\theta})$:

$$J(\boldsymbol{\theta}) = \frac{1}{n} \sum_{i=1}^{n} L\big(f(\boldsymbol{x}_i;\ \boldsymbol{\theta}),\ y_i\big)$$

3. **The Optimization:** "Learning" is the process of finding the knob settings $\boldsymbol{\theta}^{*}$ that make the average loss as small as possible:

$$\boldsymbol{\theta}^{*} = \arg\min_{\boldsymbol{\theta}} \; J(\boldsymbol{\theta})$$

Here $\arg\min$ means *"the parameter value $\boldsymbol{\theta}$ at which the function $J(\boldsymbol{\theta})$ attains its minimum"*—like carrying a lantern down a misty valley until you reach the lowest basin.

> **Model, Loss, Optimization—the three pillars of machine learning.** Nearly every parametric algorithm across these 108 lectures answers three questions: *What does $f$ look like? How is $L$ measured? And how do we search for $\boldsymbol{\theta}^{*}$?*

---

## 1.3 Three Fundamental Paradigms of Learning

Depending on the form of experience $E$ available, machine learning is traditionally divided into three main paradigms (plus a modern fourth):

### 1. Supervised Learning: Doing a Workbook with an Answer Key

Every example in the dataset comes paired with a ground-truth **label** $y_i$:

$$D = \{(\boldsymbol{x}_i,\ y_i)\}_{i=1}^{n}$$

Supervised learning splits into two core tasks based on a single question—**is the answer "which category" or "how much"?**

- **Classification ("Which category?"):** The target $y$ is one of a finite set of classes—spam vs. normal, cat vs. dog, loan default vs. repayment.
- **Regression ("How much?"):** The target $y$ is a continuous number—the selling price of a house, tomorrow's temperature, next quarter's revenue.

### 2. Unsupervised Learning: Organizing a Box of Unlabeled Old Photos

The dataset contains only inputs $\boldsymbol{x}_i$, with no target labels attached:

$$D = \{\boldsymbol{x}_i\}_{i=1}^{n}$$

Imagine being handed a box of hundreds of old photographs with no captions. As you flip through them, you naturally sort them into piles—portraits, mountain landscapes, street scenes—purely by visual similarity. Key unsupervised tasks include:

- **Clustering:** Grouping similar examples together (e.g., customer segmentation).
- **Dimensionality Reduction:** Compressing high-dimensional features into a few informative coordinates.
- **Anomaly Detection:** Spotting rare examples that deviate sharply from the norm (e.g., fraudulent transactions).

### 3. Reinforcement Learning: Learning to Ride a Bicycle Through Trial and Error

Instead of a static table of data, an **agent** interacts with an **environment**, taking actions and receiving positive or negative **rewards** $r_t$. Its goal is to learn a policy that maximizes cumulative long-term reward:

$$\max \;\; \mathbb{E}\left[\, \sum_{t=0}^{\infty} \gamma^{t}\, r_t \,\right]$$

No one can hand you a written rulebook for balancing a bicycle. You wobble, tilt, feel the feedback of gravity, adjust the handlebars, and gradually learn which actions lead to a steady ride.

### 4. A Note on Large Language Models: Self-Supervised Learning

Where do modern Large Language Models (LLMs) fit? During pre-training, they rely on **Self-Supervised Learning**: given a passage of text, the system masks or hides the next token and trains the model to predict it, using the actual next token in the original text as the ground-truth label. The supervision signal comes from the structure of the data itself, eliminating the bottleneck of manual human labeling.

| Paradigm | Data Format | What It Learns | Everyday Analogy |
|---|---|---|---|
| **Supervised Learning** | Input $\boldsymbol{x}$ + Label $y$ | Mapping from input to target | Workbook with an answer key |
| **Unsupervised Learning** | Input $\boldsymbol{x}$ only | Internal structure / groupings | Sorting unlabeled old photos |
| **Reinforcement Learning** | States, actions, rewards | Policy for long-term reward | Learning to ride a bicycle |
| **Self-Supervised Learning** | Unlabeled text / media | Predicting masked parts from context | Covering the next word on a scroll and guessing it |

---

## 1.4 Training Your First Model: Fisher's Iris & K-Nearest Neighbors

Let us move from concepts to action and build our first working classifier.

### The Data: 150 Iris Flowers

Introduced by statistician **Ronald Fisher** in 1936, the **Iris dataset** contains 150 flowers across 3 species (*Setosa*, *Versicolor*, *Virginica*—50 flowers each). For every flower, four measurements (in centimeters) were recorded:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

Each flower is a 4-dimensional feature vector $\boldsymbol{x} \in \mathbb{R}^4$, and its species is the label $y \in \{0, 1, 2\}$.

### The Algorithm: K-Nearest Neighbors (KNN)

Instead of tuning parametric knobs $\boldsymbol{\theta}$, **K-Nearest Neighbors (KNN)** follows a disarmingly simple intuition: **to classify a new flower, find the $K$ closest flowers in the training set and let them vote.**

How do we measure "closest"? Using **Euclidean distance** across all $m$ features:

$$d(\boldsymbol{x}, \boldsymbol{z}) = \sqrt{\sum_{j=1}^{m} (x_j - z_j)^2}$$

### Working It Out by Hand on Paper

Suppose our 2D training set has five known points—three Celadon (Teal: $A, B, E$) and two Amber (Gold: $C, D$):

- $A = (1, 1)$ [Celadon], $B = (2, 2)$ [Celadon], $E = (1, 3)$ [Celadon]
- $C = (3, 4)$ [Amber], $D = (5, 5)$ [Amber]

Now a new query point lands at $Q = (3, 3)$. Let us compute its Euclidean distance to each point:

$$\begin{aligned}
d(Q, C) &= \sqrt{(3-3)^2 + (3-4)^2} = 1.00 \quad \text{(Amber)} \\
d(Q, B) &= \sqrt{(3-2)^2 + (3-2)^2} = \sqrt{2} \approx 1.41 \quad \text{(Celadon)} \\
d(Q, E) &= \sqrt{(3-1)^2 + (3-3)^2} = 2.00 \quad \text{(Celadon)} \\
d(Q, A) &= d(Q, D) = \sqrt{8} \approx 2.83
\end{aligned}$$

- **If $K = 1$:** We listen only to the single nearest neighbor $C$ ($d = 1.00$). Prediction: **Amber**.
- **If $K = 3$:** The three nearest neighbors are $C$ (Amber), $B$ (Celadon), and $E$ (Celadon)—1 vote for Amber, 2 votes for Celadon. Prediction: **Celadon**.

Notice that $Q$ never moved, yet changing $K$ flipped the prediction! Because $K$ is not learned from the training data itself, but chosen beforehand by us, it is called a **hyperparameter**.

### Ten Lines of Python Code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Load data: X is a 150x4 feature matrix, y holds the 150 species labels
X, y = load_iris(return_X_y=True)

# 2. Split data: 105 flowers for training, 45 locked in a drawer for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Choose K = 3 and fit the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. Evaluate on the unseen test set and predict a new flower
print("Test accuracy:", model.score(X_test, y_test))
print("Predicted species:", model.predict([[6.1, 2.8, 4.7, 1.2]]))
```

Output:

```text
Test accuracy: 1.0
Predicted species: [1]
```

All 45 held-out test flowers are classified correctly on this split, and the new flower `[6.1, 2.8, 4.7, 1.2]` is predicted as class `1` (*Iris versicolor*).

---

## 1.5 Learning Is Not Memorizing: Fitting vs. Generalization

Why did we lock 30% of the flowers away in a drawer before training?

Imagine two students preparing for an exam. The first student memorizes the exact answer key to every problem in the workbook; the second student works out the underlying principles on scratch paper. If the exam repeats the workbook verbatim, both score 100%. Swap in new numbers, however, and only the second student succeeds.

- **Fitting:** Performing well on the training data you have already seen.
- **Generalization:** Performing well on new, unseen data.
- **Overfitting:** Memorizing the quirks and noise of the training data so rigidly that performance collapses on new data.
- **Underfitting:** Using a model too crude to capture even the basic underlying pattern.

And remember the strict rule of the locked drawer: **you must never peek at the test set to pick your hyperparameter $K$**—model selection belongs to a validation set or cross-validation, so that the final test score retains the honest weight of the unknown.

> **Fitting is the means; generalization is the end.**

---

## Key Takeaways

1. **Traditional Programming vs. Machine Learning:** Humans writing explicit rules vs. machines learning models from examples.
2. **Mitchell's Definition (1997):** A program learns from experience $E$ with respect to task $T$ and performance measure $P$ if its performance on $T$ measured by $P$ improves with $E$.
3. **The Three Pillars:** Model $f(\boldsymbol{x}; \boldsymbol{\theta})$, Loss $L(\hat{y}, y)$, and Optimization $\arg\min_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$.
4. **Paradigms:** Supervised (labeled: classification & regression), Unsupervised (unlabeled: clustering, dimensionality reduction), Reinforcement (actions & long-term rewards), and Self-Supervised (predicting masked context).
5. **Generalization:** We connect ink dots into constellations not to admire the lines on yesterday's paper, but to see whether the constellation holds when the next star appears.
