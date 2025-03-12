#import "@preview/polylux:0.3.1": *

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
  = Week 4 Discussion 1A
  #v(2em)

  Joe Lin \
  _Learning Assistant_

  October 25, 2024
]

#focus-slide[
  === Assignment 1 Review
]

#slide[
  = Core Ideas

  #text(red)[Problem: Given an image, predict which class it belongs to.]

  #pause

  #text(green)[Solution 1: *Linear Regression*, which aims to fit the observed data $(x, y)$ with a linear model $-> y = x W^T + b$.]

  #pause
  
  Dimension Check
  - What are the dimensions of $x$? #pause #text(blue)[$RR^(N times D)$]
  #pause
  - What are the dimensions of $y$? #pause #text(blue)[$RR^(N times 1)$]
  #pause
  - What are the dimensions of $W$? #pause #text(blue)[$RR^(1 times D)$]

  #pause
  
  Weaknesses? #uncover("10-")[#text(blue)[Discrete classes, but we predict all real values.]]
]

#slide[
  = Core Ideas

  #text(green)[Solution 2: *Logistic Regression* uses logistic (sigmoid) function to fit observed data $(x, y) -> y = sigma(x W^T + b)$.]

  #pause
  
  Recall the formula for logistic function. #pause #text(blue)[$sigma(z) = 1 / (1 + e^(-z))$]

  #pause

  Dimension Check
  - What are the dimensions of $y$? #pause #text(blue)[$RR^(N times C)$]
  #pause
  - What are the dimensions of $W$? #pause #text(blue)[$RR^(C times D)$]

  #pause

  Weaknesses? #uncover("9-")[#text(blue)[Predictions do not form a probability distribution over the classes.]]
]

#slide[
  = Core Ideas

  #text(green)[Solution 3: *Softmax Regression* uses softmax function to fit observed data $(x, y) -> y = "softmax"(x W^T + b)$.]

  Recall the formula for softmax function. #pause #text(blue)[$"softmax"(x) = e^(z) / (sum_(j = 1)^C e^(z_j))$]

  #pause
  
  Dimension Check
  - What are the dimensions of $y$? #pause #text(blue)[$RR^(N times C)$]
]

#slide[
  = Implementation

  ```py self.W = torch.zeros(..., requires_grad=True).to(device)```

  #pause

  What happens when we try to backpropagate and compute gradients with respect to $W$ ($(diff cal(L)) / (diff W)$)? #uncover("3-")[#text(blue)[Unable to access gradients because ```py self.W``` is not a leaf tensor]]

  #pause

  How do we fix this? \
  #uncover("4-")[```py self.W = torch.zeros(..., requires_grad=True, device=device)```]
]

#slide[
  = Implementation

  ```py cross_entropy_loss = -torch.sum(y * torch.log(p), dim=-1)```
  
  #pause

  Recall the cross entropy formula. #uncover("2-")[#text(blue)[$-log p_y$, where $p_y$ is the predicted probability that image belongs to the ground truth class]]
  
  #pause

  $ y = mat(0, 0, 1, 0, 0) \
  p = mat(0.2, 0.1, 0.4, 0.2, 0.1) \
  cal(L)_"ce" = -(0 dot log 0.2 + 0 dot log 0.1 + 1 dot log 0.4 + 0 dot log 0.2 + 0 dot log 0.1) $

  What issues may arise with this implementation?
]

#slide[
  = Implementation
  
  #block(height: 50%)[
    #grid(columns: 2, gutter: 1em,
      image("dog.jpg"),
      [
        At start of training, what would $p$ likely be? \
        #pause
        #text(blue)[$p = (0.2, 0.2, 0.2, 0.2, 0.2)$]

        #pause
        How about after training for a while? \
        #pause
        #text(blue)[$p = (0.001, 0, 0.99, 0.005, 0.004)$]

        #pause
        Why is this troublesome? #uncover("6-")[#text(blue)[$log 0$ is undefined]]
      ]
    )
  ]

  How can we fix this? \
  #uncover(7)[```py cross_entropy_loss = -torch.log(torch.sum(y * p, dim=-1))```]
]

#focus-slide[
  === Assignment 2 Preview
]

#slide[
  = Convolutions

  What are some configurable hyperparameters of convolutions? #uncover("2-")[#text(blue)[Kernel size, padding, stride, ...]]

  #pause

  Why use convolutions? #uncover("3-")[#text(blue)[Shared parameters, computational efficiency, takes advantage of inherent structure of data (for smaller datasets and model size)]]

  #pause

  Let's practice.
]