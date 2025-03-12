#import "@preview/polylux:0.3.1": *
#import "@preview/fletcher:0.4.5" as fletcher: node, edge
#import "@preview/cetz:0.3.0": *
#import "@preview/suiji:0.3.0": *
// #import "@preview/cetz-plot:0.1.0": plot

#import themes.simple: *
#import themes.university: *

#set text(font: "PT Sans")
#set enum(numbering: "1.a.i.")

#show: university-theme.with(
  color-a: rgb(39, 116, 174),
  color-b: rgb(255, 209, 0),
  color-c: rgb(255, 255, 255)
)


#matrix-slide[
  = Week 5 Discussion 1A
  #v(2em)

  Joe Lin and Krystof Latka \
  _Learning Assistants_

  November 1, 2024
]

#focus-slide[
  === Midway Review
]

#slide[
  = K-Nearest Neighbors

  What is true about the K-Nearest Neighbor (KNN) classifier? #uncover("2-")[
    #enum(numbering: "a.",
      [The L1 distance metric typically produces a smooth decision boundary, while the L2 distance metric is sharp and angled.],
      [It needs to be trained for a long time, but testing is very fast.],
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        We can tune for the most optimal value of k for a given dataset.
      ])],
      [It is an unsupervised learning algorithm.]
    )
  ]
]

#slide[
  = Regression

  Which of the following statements is correct regarding the loss functions used in Linear, Logistic, and Softmax Regression tasks? #uncover("2-")[
    #enum(numbering: "a.",
      [Linear Regression uses mean squared error (MSE), Logistic Regression uses cross entropy loss (CE), and Softmax Regression uses mean absolute error (MAE).],
      [Linear Regression uses MSE, Logistic Regression uses hinge loss, and Softmax Regression uses CE.],
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        Linear Regression uses MSE, Logistic Regression uses binary cross \ entropy loss (BCE), and Softmax Regression uses CE.
      ])],
      [All three use CE.]
    )
  ]
]

#slide[
  = Gradient Descent

  #text(red)[Problem: With naive *stochastic gradient descent*, optimization can get stuck at local minima.]

  #pause

  #text(green)[Solution: Use a running mean of gradients to build up *momentum* in a general direction.]
  $ v_(t + 1) = rho v_t + gradient f(x_t) \
  x_(t + 1) = x_t - alpha v_(t + 1) $
]

#slide[
  = Gradient Descent

  Which of the following are true about the training loss curves below?

  #align(center + horizon)[
    #grid(columns: (auto, auto), gutter: 2em)[
      #canvas(length: 3cm, {
        import draw: *

        set-style(
          mark: (fill: black, scale: 2),
          stroke: (thickness: 0.4pt, cap: "round"),
          content: (padding: 1pt)
        )

        line((-0.5, 0), (2.5, 0), mark: (end: "stealth"))
        content((1.25, -0.25), text(18pt)[num iteration])
        line((0, -0.5), (0, 1.5), mark: (end: "stealth"))
        // content((), $ cal(L) $, anchor: "south")


        line(..(for z in range(10, 111) {
          let p = (4 / calc.pow(z, 1 / 2))
          ((z / 50, p),)
        }), stroke: red + 2pt)
        content((2.5, 0.45), text(red, 18pt)[$cal(L)(w)$])
      })
    ][
      #canvas(length: 3cm, {
        import draw: *

        set-style(
          mark: (fill: black, scale: 2),
          stroke: (thickness: 0.4pt, cap: "round"),
          content: (padding: 1pt)
        )

        line((-0.5, 0), (2.5, 0), mark: (end: "stealth"))
        content((1.25, -0.25), text(18pt)[num iteration])
        line((0, -0.5), (0, 1.5), mark: (end: "stealth"))
        // content((), $ cal(L) $, anchor: "south")

        let rng = gen-rng(42)
        let v = ()
        (rng, v) = uniform(rng, low: -0.1, high: 0.1, size: 102)
        line(..(for z in range(10, 111) {
          let p = (4 / calc.pow(z, 1 / 2))
          ((z / 50, p + v.at(z - 10)),)
        }), stroke: red + 2pt)
        content((2.5, 0.45), text(red, 18pt)[$cal(L)(w)$])
      })
    ]
  ]
]

#slide[
  = Gradient Descent

  #enum(numbering: "a.",
    [Graph A is *Mini-Batch Gradient Descent* -- gradient estimates using batch and minibatch are equivalent.],
    [#alternatives-cases(("1", "2"), case => [
      #set text(fill: blue) if case == 1
      Graph A is *Batch Gradient Descent* -- accurate gradient estimates, \ resulting in a decrease in loss at every iteration.
    ])],
    [#alternatives-cases(("1", "2"), case => [
      #set text(fill: blue) if case == 1
      Graph B is *Mini-Batch Gradient Descent* -- noisy gradient estimates, \ resulting in oscillations in the loss trajectory.
    ])],
    [Graph B is *Batch Gradient Descent* -- considering more data points, so introduces more noise to gradient calculation.]
  )
]

#slide[
  = Regularization

  Which of the following are valid techniques to perform regularization? #uncover("2-")[
    #enum(numbering: "a.",
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        Data Augmentation
      ])],
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        Norm Penalties
      ])],
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        Model Ensembling
      ])],
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        Dropout
      ])]
    )
  ]
]

#slide[
  = Hyperparameters

  You are tasked with training a *Fully Connected Neural Network* with batch normalization and dropout. Which of the following are *hyperparameters*? #uncover("2-")[
    #enum(numbering: "a.",
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        Loss Function $cal(L)$
      ])],
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        Dropout Probability $p$
      ])],
      [Batch Normalization's $gamma$ and $beta$],
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        Batch Size $B$
      ])]
    )
  ]

]

#slide[
  = Universal Approximation Theorem

  What does this claim? #uncover("2-")[#text(blue)[With enough neurons, a neural network can approximate any function $f$.]]

  #pause

  #text(red)[However, this does not mean a neural network can efficiently learn such an approximation.]

  #align(center)[#image("uat.png", height: 40%)]
]

#slide[
  = Neural Networks

  #grid(columns: (auto, auto), )[
    A *Neural Network* is a computational model that makes decisions and predictions in a way inspired by biological systems.

    Generally consists of:
    - Input $x$
    - Hidden Layer(s) $h_i$
    - Output Layer $y$
    - Activation Functions $phi(x)$
  ][
    #align(center)[
      #fletcher.diagram(
        node-inset: 15pt,
        {
          node($x$, enclose: ((-1.2, 1), (-1, -1)), shape: "rect", stroke: green + 0.5pt, fill: green.lighten(80%))
          node($h$, enclose: ((-0.1, 1.5), (0.1, -1.5)), shape: "rect", stroke: red + 0.5pt, fill: red.lighten(80%))
          node($y$, enclose: ((1, 0.25), (1.2, -0.25)), shape: "rect", stroke: blue + 0.5pt, fill: blue.lighten(80%))
          edge((-1.1, 0), (0, 0), $W_1$, "-|>")
          edge((0, 0), (1.1, 0), $W_2$, "-|>")
        }
      )
    ]
  ]
]

#slide[
  = Neural Networks
  
  Suppose we have a neural network $F: RR^(N times (10 dot 10)) -> RR^(N times 1)$ that predicts a class based on a grayscale image.

  ```py
  F = nn.Sequential(
    nn.Linear(100, 10, bias=False),
    nn.ReLU(),
    nn.BatchNorm(10),
    nn.Linear(10, 1, bias=True)
  )
  ```

  How many *trainable parameters* does $F$ have? #pause #text(blue)[$100 dot 10 + 2 dot 10 + 10 dot 1 + 1 = 1031$]
]

#slide[
  = Activation Functions

  What are desirable properties of activation functions? #uncover("2-")[#text(blue)[Non-linear, differentiable, supplies non-vanishing gradients, ...]]

  #pause

  Why do we need non-linearity anyways? #uncover("3-")[#text(blue)[
    Suppose we had a linear activation $phi$. You can think of this as another matrix.
    $ y &= W_2 phi(W_1 x + b_1) + b_2 \
    &= W_2 phi W_1 x + W_2 phi b_1 + b_2 \
    &= (phi W_2 W_1) x + (phi W_2 b_1 + b_2) $

    Same as $y = W_3 x + b_3$, where $W_3 = phi W_2 W_1$ and $b_3 = phi W_2 b_1 + b_2$.
  ]]
]

#slide[
  = Activation Functions

  #grid(columns: (auto, auto), gutter: 2em)[
    #canvas(length: 3cm, {
      import draw: *

      set-style(
        mark: (fill: black, scale: 2),
        stroke: (thickness: 0.4pt, cap: "round"),
        content: (padding: 1pt)
      )

      grid((-1.5, -1.5), (1.5, 1.5), step: 0.5, stroke: gray + 0.2pt)

      line((-1.5, 0), (1.5, 0), mark: (end: "stealth"))
      content((), $ z $, anchor: "west")
      line((0, -1.5), (0, 1.5), mark: (end: "stealth"))
      content((), $ phi $, anchor: "south")


      line(..(for z in range(-150, 151) {
        let p = ((calc.exp(2 * z / 20) - 1) / (calc.exp(2 * z / 20) + 1))
        ((z / 100, p),)
      }), stroke: red + 2pt, mark: (start: "straight", end: "straight"))
    })
  ][
    - Name: #uncover("2-")[#text(blue)[tanh]]
    - Formula: #uncover("3-")[#text(blue)[$phi(z) = (e^(2x) - 1) / (e^(2x) + 1)$]]
    - Properties: #uncover("4-")[#text(blue)[Non-linear, differentiable everywhere, $gradient approx 0$ when $z -> plus.minus infinity$]]
  ]
]

#slide[
  = Activation Functions

  #grid(columns: (auto, auto), gutter: 2em)[
    #canvas(length: 3cm, {
      import draw: *

      set-style(
        mark: (fill: black, scale: 2),
        stroke: (thickness: 0.4pt, cap: "round"),
        content: (padding: 1pt)
      )

      grid((-1.5, -1.5), (1.5, 1.5), step: 0.5, stroke: gray + 0.2pt)

      line((-1.5, 0), (1.5, 0), mark: (end: "stealth"))
      content((), $ z $, anchor: "west")
      line((0, -1.5), (0, 1.5), mark: (end: "stealth"))
      content((), $ phi $, anchor: "south")


      line((-1.5, 0), (0, 0), stroke: red + 2pt, mark: (start: "straight"))
      line((0, 0), (1.5, 1.5), stroke: red + 2pt, mark: (end: "straight"))
    })
  ][
    - Name: #uncover("2-")[#text(blue)[Rectified Linear Unit (ReLU)]]
    - Formula: #uncover("3-")[#text(blue)[$phi(z) = max(0, z)$]]
    - Properties: #uncover("4-")[#text(blue)[Non-linear, differentiable everywhere except $x = 0$, $gradient > 0$ when $z > 0$]]
  ]
]

#slide[
  = Backpropagation

  Helpful gradient gates to remember!
  
  #pause
  #align(center)[
    #grid(columns: (auto, auto), gutter: 2em, align: center)[
      *Add Gate* \
      #fletcher.diagram(
        node-inset: 15pt,
        {
          node((1.5, 0), $+$, stroke: black + 0.5pt)
          edge((0, 0.5), (1, 0.5), (1.5, 0), "-|>")
          edge((0, -0.5), (1, -0.5), (1.5, 0), "-|>")
          node((0.5, 0.3), $3$, inset: 0pt)
          node((0.5, 0.7), uncover("5-")[#text(blue)[$2$]], inset: 0pt)
          node((0.5, -0.7), $2$, inset: 0pt)
          node((0.5, -0.3), uncover("4-")[#text(blue)[$2$]], inset: 0pt)
          edge((1.5, 0), (3, 0), "-|>")
          node((2.25, -0.2), $5$, inset: 0pt)
          node((2.25, 0.2), uncover("3-")[#text(blue)[$2$]], inset: 0pt)
        }
      )

      #uncover("6-")[#text(blue)[Gradient Distributor]]
    ][
      #uncover("7-")[
        *Copy Gate* \
        #fletcher.diagram(
          node-inset: 15pt,
          {
            node((1.5, 0), text(rgb(0, 0, 0, 0))[$+$], stroke: black + 0.5pt)
            edge((0, 0), (1.5, 0), "-|>")
            node((0.5, -0.2), $4$, inset: 0pt)
            node((0.5, 0.2), uncover("9-")[#text(blue)[$5$]], inset: 0pt)

            edge((1.5, 0), (2, 0.5), (3, 0.5), "-|>")
            edge((1.5, 0), (2, -0.5), (3, -0.5), "-|>")
            node((2.5, 0.3), $4$, inset: 0pt)
            node((2.5, 0.7), uncover("8-")[#text(blue)[$3$]], inset: 0pt)
            node((2.5, -0.7), $4$, inset: 0pt)
            node((2.5, -0.3), uncover("8-")[#text(blue)[$2$]], inset: 0pt)
          }
        )
      ]

      #uncover("10-")[#text(blue)[Gradient Adder]]
    ]
  ]
]

#slide[
  = Backpropagation

  Helpful gradient gates to remember!
  
  #pause
  #align(center)[
    #grid(columns: (auto, auto), gutter: 2em, align: center)[
      *Multiply Gate* \
      #fletcher.diagram(
        node-inset: 15pt,
        {
          node((1.5, 0), $times$, stroke: black + 0.5pt)
          edge((0, 0.5), (1, 0.5), (1.5, 0), "-|>")
          edge((0, -0.5), (1, -0.5), (1.5, 0), "-|>")
          node((0.5, 0.3), $2$, inset: 0pt)
          node((0.5, 0.7), uncover("5-")[#text(blue)[$5 dot 3 = 15$]], inset: 0pt)
          node((0.5, -0.7), $3$, inset: 0pt)
          node((0.5, -0.3), uncover("4-")[#text(blue)[$5 dot 2 = 10$]], inset: 0pt)
          edge((1.5, 0), (3, 0), "-|>")
          node((2.25, -0.2), $6$, inset: 0pt)
          node((2.25, 0.2), uncover("3-")[#text(blue)[$5$]], inset: 0pt)
        }
      )

      #uncover("6-")[#text(blue)[Swap Multiplier]]
    ][
      #uncover("7-")[
        *Max Gate* \
        #fletcher.diagram(
          node-inset: 15pt,
          {
            node((1.5, 0), text(18pt)[max], stroke: black + 0.5pt, shape: "circle")
            edge((0, 0.5), (1, 0.5), (1.5, 0), "-|>")
            edge((0, -0.5), (1, -0.5), (1.5, 0), "-|>")
            node((0.5, 0.3), $4$, inset: 0pt)
            node((0.5, 0.7), uncover("5-")[#text(blue)[$0$]], inset: 0pt)
            node((0.5, -0.7), $5$, inset: 0pt)
            node((0.5, -0.3), uncover("4-")[#text(blue)[$9$]], inset: 0pt)
            edge((1.5, 0), (3, 0), "-|>")
            node((2.25, -0.2), $5$, inset: 0pt)
            node((2.25, 0.2), uncover("3-")[#text(blue)[$9$]], inset: 0pt)
          }
        )
      ]

      #uncover("10-")[#text(blue)[Gradient Router]]
    ]
  ]
]

#slide[
  = Backpropagation

  Compute the forward and backward pass. $phi$ is a ReLU activation.
  #fletcher.diagram(
    node-inset: 15pt,
    {
      node((1, 0), $times$, stroke: black + 0.5pt)
      node((1.9, 0.5), $+$, stroke: black + 0.5pt)
      node((2.8, 0.5), $sigma$, stroke: black + 0.5pt)
      node((3.7, 1), $times$, stroke: black + 0.5pt)
      node((4.6, 1), $+$, stroke: black + 0.5pt)
      node((5.5, 1), $phi$, stroke: black + 0.5pt)

      edge((0, 0.25), (0.5, 0.25), (1, 0), "-|>")
      edge((0, -0.35), (0.5, -0.35), (1, 0), "-|>")
      node((0, 0.25), $W_0$, inset: 0pt)
      node((0.35, 0.1), uncover("1-")[$2$], inset: 0pt)
      node((0, -0.35), $x$, inset: 0pt)
      node((0.35, -0.5), uncover("1-")[$2$], inset: 0pt)

      edge((1, 0), (1.6, 0), (1.9, 0.5), "-|>")
      node((1.4, -0.2), uncover("2-")[$4$], inset: 0pt)
      edge((0, 0.75), (1.6, 0.75), (1.9, 0.5), "-|>")
      node((0, 0.75), $W_1$, inset: 0pt)
      node((0.9, 0.6), uncover("1-")[$-4$], inset: 0pt)

      edge((1.9, 0.5), (2.8, 0.5), "-|>")
      node((2.35, 0.3), uncover("3-")[$0$], inset: 0pt)

      edge((2.8, 0.5), (3.4, 0.5), (3.7, 1), "-|>")
      node((3.2, 0.3), uncover("4-")[$0.5$], inset: 0pt)
      edge((0, 1.25), (3.4, 1.25), (3.7, 1), "-|>")
      node((0, 1.25), $W_2$, inset: 0pt)
      node((1.6, 1.1), uncover("1-")[$4$], inset: 0pt)

      edge((3.7, 1), (4.6, 1), "-|>")
      node((4.15, 0.8), uncover("5-")[$2$], inset: 0pt)
      edge((0, 1.75), (4.3, 1.75), (4.6, 1), "-|>")
      node((0, 1.75), $W_3$, inset: 0pt)
      node((2.1, 1.6), uncover("1-")[$5$], inset: 0pt)

      edge((4.6, 1), (5.5, 1), "-|>")
      node((5.05, 0.8), uncover("6-")[$7$], inset: 0pt)
      
      edge((5.5, 1), (6.5, 1), "-|>")
      node((6, 0.8), uncover("7-")[$7$], inset: 0pt)
      node((6, 1.2), uncover("7-")[#text(blue)[$1$]], inset: 0pt)
    }
  )
]

#slide[
  = Backpropagation

  Compute the forward and backward pass. $phi$ is a ReLU activation.
  #fletcher.diagram(
    node-inset: 15pt,
    {
      node((1, 0), $times$, stroke: black + 0.5pt)
      node((1.9, 0.5), $+$, stroke: black + 0.5pt)
      node((2.8, 0.5), $sigma$, stroke: black + 0.5pt)
      node((3.7, 1), $times$, stroke: black + 0.5pt)
      node((4.6, 1), $+$, stroke: black + 0.5pt)
      node((5.5, 1), $phi$, stroke: black + 0.5pt)

      edge((0, 0.25), (0.5, 0.25), (1, 0), "-|>")
      edge((0, -0.35), (0.5, -0.35), (1, 0), "-|>")
      node((0, 0.25), $W_0$, inset: 0pt)
      node((0.35, 0.1), uncover("1-")[$2$], inset: 0pt)
      node((0.35, 0.4), uncover("6-")[#text(blue)[$2$]], inset: 0pt)
      node((0, -0.35), $x$, inset: 0pt)
      node((0.35, -0.5), uncover("1-")[$2$], inset: 0pt)
      node((0.35, -0.2), uncover("6-")[#text(blue)[$2$]], inset: 0pt)

      edge((1, 0), (1.6, 0), (1.9, 0.5), "-|>")
      node((1.4, -0.2), uncover("1-")[$4$], inset: 0pt)
      node((1.4, 0.2), uncover("5-")[#text(blue)[$1$]], inset: 0pt)
      edge((0, 0.75), (1.6, 0.75), (1.9, 0.5), "-|>")
      node((0, 0.75), $W_1$, inset: 0pt)
      node((0.9, 0.6), uncover("1-")[$-4$], inset: 0pt)
      node((0.9, 0.9), uncover("5-")[#text(blue)[$1$]], inset: 0pt)

      edge((1.9, 0.5), (2.8, 0.5), "-|>")
      node((2.35, 0.3), uncover("1-")[$0$], inset: 0pt)
      node((2.35, 0.7), uncover("4-")[#text(blue)[$1$]], inset: 0pt)

      edge((2.8, 0.5), (3.4, 0.5), (3.7, 1), "-|>")
      node((3.2, 0.3), uncover("1-")[$0.5$], inset: 0pt)
      node((3.2, 0.7), uncover("3-")[#text(blue)[$4$]], inset: 0pt)
      edge((0, 1.25), (3.4, 1.25), (3.7, 1), "-|>")
      node((0, 1.25), $W_2$, inset: 0pt)
      node((1.6, 1.1), uncover("1-")[$4$], inset: 0pt)
      node((1.6, 1.4), uncover("3-")[#text(blue)[$0.5$]], inset: 0pt)

      edge((3.7, 1), (4.6, 1), "-|>")
      node((4.15, 0.8), uncover("1-")[$2$], inset: 0pt)
      node((4.15, 1.2), uncover("2-")[#text(blue)[$1$]], inset: 0pt)
      edge((0, 1.75), (4.3, 1.75), (4.6, 1), "-|>")
      node((0, 1.75), $W_3$, inset: 0pt)
      node((2.1, 1.6), uncover("1-")[$5$], inset: 0pt)
      node((2.1, 1.9), uncover("2-")[#text(blue)[$1$]], inset: 0pt)

      edge((4.6, 1), (5.5, 1), "-|>")
      node((5.05, 0.8), uncover("1-")[$7$], inset: 0pt)
      node((5.05, 1.2), uncover("1-")[#text(blue)[$1$]], inset: 0pt)
      
      edge((5.5, 1), (6.5, 1), "-|>")
      node((6, 0.8), uncover("1-")[$7$], inset: 0pt)
      node((6, 1.2), uncover("1-")[#text(blue)[$1$]], inset: 0pt)
    }
  )
]

#slide[
  = Normalization

  #text(red)[Problem: Input features can have different scales, impacting the efficiency and performance of training.]

  #pause

  #text(green)[Solution: *Input Normalization* can be applied on the network inputs using dataset statistics.]

  #pause

  $ hat(mu)_j = 1 / N sum_(i = 1)^N x_(i, j) \
  sigma^2_j = 1 / N sum_(i = 1)^N (x_(i, j) - hat(mu)_j)^2 $
]

#slide[
  = Normalization

  #text(red)[Problem: Inputs to hidden layers are not normalized and we don't have prior feature statistics.]

  #pause

  #text(green)[Solution: Aggregate statistics to normalize and learn scale + shift parameters with *Batch Normalization*.]

  #pause

  #grid(columns: (auto, auto, auto), gutter: 2em, align: center + horizon)[
    $ mu_j, sigma_j^2 #text()[are running avg] $
  ][
    $ hat(x)_(i, j) = (x_(i, j) - mu_j) / sqrt(sigma_j^2 + epsilon) \
    y_(i, j) = gamma_j hat(x)_(i, j) + beta_j $
  ][
    At test time, use aggregated statistics from training.
  ]
]

#slide[
  = Convolutions

  Which of the following is true about the *receptive field* of a convolutional neural network as more layers are added? #uncover("2-")[
    #enum(numbering: "a.",
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        It increases and depends on the kernel size and stride of each layer.
      ])],
      [It remains the same regardless of the depth of the network.],
      [It only depends on the kernel size of the first layer.],
      [It decreases with each additional convolutional layer.]
    )
  ]
]

#slide[
  = Convolutions

  Given an input $x in RR^(H times W times C)$, where $H = W = C = 2$, compute the output of a convolution with a *kernel* $K in RR^(2 times 2 times 2)$, *padding* of $1$, and *stride* of $2$.

  #align(center + horizon)[
    #canvas(length: 3cm, {
      import draw: *

      set-style(
        mark: (fill: black, scale: 2),
        stroke: (thickness: 0.4pt, cap: "round"),
        content: (padding: 1pt)
      )

      let coords = ((-0.25, 0.25), (0.25, 0.25), (-0.25, -0.25), (0.25, -0.25))
      let inputs = (($1$, $3$, $2$, $4$), ($5$, $3$, $1$, $5$))
      for i in range(inputs.len()) {
        rect((-0.5 + i / 4, -0.5 - i / 4), (0.5 + i / 4, 0.5 - i / 4), fill: gray.lighten(70%))
        grid((-0.5 + i / 4, -0.5 - i / 4), (0.5 + i / 4, 0.5 - i / 4), step: 0.5, stroke: gray + 0.2pt)
        for j in range(coords.len()) {
          content((coords.at(j).at(0) + i / 4, coords.at(j).at(1) - i / 4), align(center + horizon)[#inputs.at(i).at(j)])
        }
      }
      content((0.125, -1.1), align(center + horizon)[Input])
      content((2.125, -1.1), align(center + horizon)[Kernel])

      let kernels = (($2$, $3$, $1$, $6$), ($4$, $8$, $2$, $5$))
      for i in range(kernels.len()) {
        rect((-0.5 + i / 4 + 2, -0.5 - i / 4), (0.5 + i / 4 + 2, 0.5 - i / 4), fill: gray.lighten(70%))
        grid((-0.5 + i / 4 + 2, -0.5 - i / 4), (0.5 + i / 4 + 2, 0.5 - i / 4), step: 0.5, stroke: gray + 0.2pt)
        for j in range(coords.len()) {
          content((coords.at(j).at(0) + i / 4 + 2, coords.at(j).at(1) - i / 4), align(center + horizon)[#kernels.at(i).at(j)])
        }
      }
    })
  ]
]

#slide[
  = Convolutions

  #canvas(length: 3cm, {
    import draw: *

    set-style(
      mark: (fill: black, scale: 2),
      stroke: (thickness: 0.4pt, cap: "round"),
      content: (padding: 1pt)
    )

    let coords = ((-0.25, 0.25), (0.25, 0.25), (-0.25, -0.25), (0.25, -0.25))
    let inputs = (($1$, $3$, $2$, $4$), ($5$, $3$, $1$, $5$))
    for i in range(inputs.len()) {
      rect((-0.5, -0.5 - 1.5 * i), (0.5, 0.5 - 1.5 * i), fill: gray.lighten(70%))
      grid((-0.5, -0.5 - 1.5 * i), (0.5, 0.5 - 1.5 * i), step: 0.5, stroke: gray + 0.2pt)
      for j in range(coords.len()) {
        content((coords.at(j).at(0), coords.at(j).at(1) - 1.5 * i), align(center + horizon)[#inputs.at(i).at(j)])
      }
    }
    content((0, -2.5), align(center + horizon)[Input])

    content((-1.75, 0), align(center + horizon)[Channel $1$])
    content((-1.75, -1.5), align(center + horizon)[Channel $2$])

    let kernels = (($2$, $3$, $1$, $6$), ($4$, $8$, $2$, $5$))
    for i in range(kernels.len()) {
      rect((-0.5 + 1.5, -0.5 - 1.5 * i), (0.5 + 1.5, 0.5 - 1.5 * i), fill: gray.lighten(70%))
      grid((-0.5 + 1.5, -0.5 - 1.5 * i), (0.5 + 1.5, 0.5 - 1.5 * i), step: 0.5, stroke: gray + 0.2pt)
      for j in range(coords.len()) {
        content((coords.at(j).at(0) + 1.5, coords.at(j).at(1) - 1.5 * i), align(center + horizon)[#kernels.at(i).at(j)])
      }
    }

    content((1.5, -2.5), align(center + horizon)[Kernel])
  })
]

#slide[
  = Convolutions

  #align(center + horizon)[
    #block(height: 80%)[
      #let all_inputs = ((($0$, $0$, $0$, $0$, $0$, $1$, $3$, $0$, $0$, $2$, $4$, $0$, $0$, $0$, $0$, $0$),), (($0$, $0$, $0$, $0$, $0$, $5$, $3$, $0$, $0$, $1$, $5$, $0$, $0$, $0$, $0$, $0$),))
      #let all_kernels = ((($2$, $3$, $1$, $6$),), (($4$, $8$, $2$, $5$),))
      #let all_outputs = ((($6$, $3$, $6$, $8$),), (($25$, $6$, $8$, $20$),))
      #for ch in range(2) {
        for c in range(4) {
          only(c + 1 + 4 * ch)[
            #canvas(length: 3cm, {
              import draw: *

              set-style(
                mark: (fill: black, scale: 2),
                stroke: (thickness: 0.4pt, cap: "round"),
                content: (padding: 1pt)
              )

              let dx = calc.rem(c, 2) * 1
              let dy = -calc.floor(c / 2) * 1

              let coords = ((-0.75, 0.75), (-0.25, 0.75), (0.25, 0.75), (0.75, 0.75), (-0.75, 0.25), (-0.25, 0.25), (0.25, 0.25), (0.75, 0.25), (-0.75, -0.25), (-0.25, -0.25), (0.25, -0.25), (0.75, -0.25), (-0.75, -0.75), (-0.25, -0.75), (0.25, -0.75), (0.75, -0.75))
              let inputs = all_inputs.at(ch)
              for i in range(inputs.len()) {
                rect((-1, -1), (1, 1), fill: gray.lighten(70%))
                grid((-1, -1), (1, 1), step: 0.5, stroke: gray + 0.2pt)
                for j in range(coords.len()) {
                  content((coords.at(j).at(0), coords.at(j).at(1) - 1.5 * i), align(center + horizon)[#inputs.at(i).at(j)])
                }
              }

              rect((-1.1 + dx, 1.1 + dy), (0.1 + dx, -0.1 + dy), stroke: blue + 2pt, radius: 12pt, fill: blue.transparentize(90%))

              let kernels = all_kernels.at(ch)
              coords = ((-0.25, 0.25), (0.25, 0.25), (-0.25, -0.25), (0.25, -0.25))
              for i in range(kernels.len()) {
                rect((-0.5 + 2, -0.5), (0.5 + 2, 0.5), fill: gray.lighten(70%))
                grid((-0.5 + 2, -0.5 - 1.5 * i), (0.5 + 2, 0.5 - 1.5 * i), step: 0.5, stroke: gray + 0.2pt)
                for j in range(coords.len()) {
                  content((coords.at(j).at(0) + 2, coords.at(j).at(1)), align(center + horizon)[#kernels.at(i).at(j)])
                }
              }

              content((3, 0), align(horizon)[$->$])

              let outputs = all_outputs.at(ch)
              for i in range(outputs.len()) {
                rect((-0.5 + 4, -0.5), (0.5 + 4, 0.5), fill: gray.lighten(70%))
                grid((-0.5 + 4, -0.5 - 1.5 * i), (0.5 + 4, 0.5 - 1.5 * i), step: 0.5, stroke: gray + 0.2pt)
                for j in range(coords.len()) {
                  if c >= j {
                    content((coords.at(j).at(0) + 4, coords.at(j).at(1)), align(center + horizon)[#outputs.at(i).at(j)])
                  }
                }
              }

              rect((-0.6 + 4 + dx / 2, 0.6 + dy / 2), (0.1 + 4 + dx / 2, -0.1 + dy / 2), stroke: blue + 2pt, radius: 12pt, fill: blue.transparentize(90%))
            })
          ]
        }
      }
    ]
  ]
]

#slide[
  = Convolutions

  Which operation reduces feature map size the most? #uncover("2-")[
    #enum(numbering: "a.",
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        Global Average Pooling
      ])],
      [$2 times 2$ Max Pooling with stride of $3$],
      [Softmax activation],
      [$5 times 5$ Convolution]
    )
  ]
]

#slide[
  = Convolutions

  A convolution layer has $16$ *filters* $K in RR^(7 times 7 times 8)$ (note: $RR^(H times W times C)$), padding of $2$, and stride of $3$. Compute the number of parameters, including bias.

  #pause

  #text(blue)[
    $ 16 (7 dot 7 dot 8 + 1) = 16 dot 7 dot 7 dot 8 + 16 = 6288 $
  ]

  #pause
  
  State the dimension of output $y$ for an input $x in RR^(32 times 32 times 8)$?

  #pause

  #text(blue)[
    $ H_"out" = floor((H_"in" - k + 2p) / s) + 1 = 10, W_"out" = floor((W_"in" - k + 2p) / s) + 1 = 10 \ C_"out" = 16 $
  ]
]

#slide[
  = Convolutions

  Which of the following is true about convolutions? #uncover("2-")[
    #enum(numbering: "a.",
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        The number of biases is equal to the number of filters.
      ])],
      [Training a convolutional neural network (CNN) involves no *inductive bias*.],
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        The number of output channels is equal to the number of filters.
      ])],
      [#alternatives-cases(("1, 2", "3"), case => [
        #set text(fill: blue) if case == 1
        *Dilated convolutions* increase the receptive field.
      ])]
    )
  ]
]